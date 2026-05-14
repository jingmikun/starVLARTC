from __future__ import annotations

import math

import torch


def get_prefix_weights_torch(
    inference_delay: int,
    execution_horizon: int,
    chunk_size: int,
    schedule: str = "exp",
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Return [H] weights.
    i < inference_delay: weight 1
    inference_delay <= i < execution_horizon: decay according to schedule
    i >= execution_horizon: weight 0
    """
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be non-negative, got {chunk_size}")

    horizon = int(chunk_size)
    delay = max(0, int(inference_delay))
    exec_horizon = max(0, int(execution_horizon))
    delay = min(delay, horizon)
    exec_horizon = min(exec_horizon, horizon)

    weights = torch.zeros(horizon, device=device, dtype=dtype)
    prefix_end = min(delay, exec_horizon)
    if prefix_end > 0:
        weights[:prefix_end] = 1.0

    if exec_horizon <= delay:
        return weights

    schedule = schedule.lower()
    tail_len = exec_horizon - delay
    tail_idx = torch.arange(tail_len, device=device, dtype=dtype)

    if schedule == "exp":
        denom = max(tail_len - 1, 1)
        tail = torch.exp(-5.0 * tail_idx / float(denom))
    elif schedule == "linear":
        tail = (tail_len - tail_idx) / float(tail_len)
    elif schedule == "ones":
        tail = torch.ones(tail_len, device=device, dtype=dtype)
    elif schedule == "zeros":
        tail = torch.zeros(tail_len, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported RTC prefix attention schedule: {schedule}")

    weights[delay:exec_horizon] = tail
    return weights


def compute_rtc_guidance_weight(
    t_cont: float,
    max_guidance_weight: float = 5.0,
    eps: float = 1e-6,
) -> float:
    """
    Compute RTC guidance clipping coefficient:
    min(beta, (1 - tau) / (tau * r_tau^2))
    """
    beta = float(max_guidance_weight)
    if beta <= 0.0:
        return 0.0

    tau = min(max(float(t_cont), eps), 1.0 - eps)
    r_tau = max(1.0 - tau, eps)
    raw = (1.0 - tau) / (tau * r_tau * r_tau)
    if not math.isfinite(raw):
        return beta
    return float(max(0.0, min(beta, raw)))


def pad_leftover_chunk(
    prev_chunk_left_over,
    *,
    batch_size: int,
    horizon: int,
    action_dim: int,
    device,
    dtype,
    name: str = "prev_chunk_left_over",
) -> tuple[torch.Tensor, int]:
    """
    Accept [R,D], [1,R,D], or [B,R,D], with R <= H.
    Return padded [B,H,D] and leftover_len R.
    """
    if prev_chunk_left_over is None:
        return torch.zeros(batch_size, horizon, action_dim, device=device, dtype=dtype), 0

    chunk = torch.as_tensor(prev_chunk_left_over, device=device, dtype=dtype)
    if chunk.ndim == 2:
        chunk = chunk.unsqueeze(0)
    elif chunk.ndim != 3:
        raise ValueError(f"{name} must have shape [R,D], [1,R,D], or [B,R,D], got {tuple(chunk.shape)}")

    if chunk.shape[-1] != action_dim:
        raise ValueError(f"{name} action dim mismatch: expected {action_dim}, got {chunk.shape[-1]}")
    if chunk.shape[1] > horizon:
        raise ValueError(f"{name} leftover length must be <= horizon {horizon}, got {chunk.shape[1]}")
    if chunk.shape[0] not in (1, batch_size):
        raise ValueError(f"{name} batch mismatch: expected batch 1 or {batch_size}, got {chunk.shape[0]}")

    leftover_len = int(chunk.shape[1])
    padded = torch.zeros(batch_size, horizon, action_dim, device=device, dtype=dtype)
    if leftover_len == 0:
        return padded, 0

    if chunk.shape[0] == 1 and batch_size != 1:
        chunk = chunk.expand(batch_size, -1, -1)
    padded[:, :leftover_len, :] = chunk
    return padded, leftover_len


def make_rtc_action_mask(
    *,
    horizon: int,
    action_dim: int,
    weights: torch.Tensor,
    leftover_len: int,
    gripper_indices: tuple[int, ...] | None = (6, 13),
    gripper_scale: float = 0.0,
) -> torch.Tensor:
    """
    Return [1,H,D].
    Mask after leftover_len should be 0.
    Gripper dims should be scaled by gripper_scale.
    """
    if weights.ndim != 1 or weights.shape[0] != horizon:
        raise ValueError(f"weights must have shape [{horizon}], got {tuple(weights.shape)}")

    valid_len = max(0, min(int(leftover_len), int(horizon)))
    mask = torch.zeros(1, horizon, action_dim, device=weights.device, dtype=weights.dtype)
    if valid_len == 0:
        return mask

    mask[:, :valid_len, :] = weights[:valid_len].view(1, valid_len, 1)
    if gripper_indices is not None:
        for idx in gripper_indices:
            if 0 <= int(idx) < action_dim:
                mask[:, :valid_len, int(idx)] *= float(gripper_scale)
    return mask
