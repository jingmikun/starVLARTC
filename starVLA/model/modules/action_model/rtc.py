"""Real-Time Chunking utilities.

RTC runs entirely in the normalized action space.  The client owns action
queueing, un-normalization, gripper binarization, and robot safety checks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Deque, Optional

import numpy as np
import torch


class RTCAttentionSchedule(str, Enum):
    LINEAR = "LINEAR"
    EXP = "EXP"
    ONES = "ONES"
    ZEROS = "ZEROS"


@dataclass
class RTCConfig:
    enabled: bool = False
    execution_horizon: int = 8
    prefix_attention_horizon: Optional[int] = None
    max_guidance_weight: float = 5.0
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self) -> None:
        self.execution_horizon = int(self.execution_horizon)
        if self.execution_horizon <= 0:
            raise ValueError("RTCConfig.execution_horizon must be positive.")

        if self.prefix_attention_horizon is not None:
            self.prefix_attention_horizon = int(self.prefix_attention_horizon)
            if self.prefix_attention_horizon < 0:
                raise ValueError("RTCConfig.prefix_attention_horizon must be non-negative.")

        self.max_guidance_weight = float(self.max_guidance_weight)
        if self.max_guidance_weight < 0:
            raise ValueError("RTCConfig.max_guidance_weight must be non-negative.")

        if not isinstance(self.prefix_attention_schedule, RTCAttentionSchedule):
            self.prefix_attention_schedule = RTCAttentionSchedule(str(self.prefix_attention_schedule).upper())

        self.debug_maxlen = int(self.debug_maxlen)
        if self.debug_maxlen <= 0:
            raise ValueError("RTCConfig.debug_maxlen must be positive.")

    @classmethod
    def from_any(cls, value: "RTCConfig | dict[str, Any] | None") -> "RTCConfig":
        if value is None:
            return cls()
        if isinstance(value, RTCConfig):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"rtc must be a dict, RTCConfig, or None; got {type(value)!r}.")

        valid_keys = {field.name for field in fields(cls)}
        kwargs = {key: value[key] for key in valid_keys if key in value}
        return cls(**kwargs)

    def to_dict(self, chunk_size: Optional[int] = None) -> dict[str, Any]:
        prefix_horizon = (
            resolve_prefix_attention_horizon(self, chunk_size)
            if chunk_size is not None
            else self.prefix_attention_horizon
        )
        return {
            "enabled": self.enabled,
            "execution_horizon": self.execution_horizon,
            "prefix_attention_horizon": prefix_horizon,
            "max_guidance_weight": self.max_guidance_weight,
            "prefix_attention_schedule": self.prefix_attention_schedule.value,
            "debug": self.debug,
            "debug_maxlen": self.debug_maxlen,
        }


def resolve_prefix_attention_horizon(rtc_config: RTCConfig, chunk_size: int) -> int:
    """Resolve RTC guidance prefix length.

    By default RTC constrains the overlap between the old chunk and the new
    chunk.  If the client executes every ``s`` steps, the overlap is ``H - s``.
    """

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if rtc_config.prefix_attention_horizon is None:
        return max(chunk_size - int(rtc_config.execution_horizon), 0)
    return min(int(rtc_config.prefix_attention_horizon), chunk_size)


def make_prefix_attention_weights(
    chunk_size: int,
    inference_delay: int,
    rtc_config: RTCConfig,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return per-timestep guidance weights shaped ``[H]``."""

    chunk_size = int(chunk_size)
    inference_delay = max(int(inference_delay), 0)
    dtype = dtype or torch.float32
    weights = torch.zeros(chunk_size, device=device, dtype=dtype)

    prefix_horizon = resolve_prefix_attention_horizon(rtc_config, chunk_size)
    if prefix_horizon == 0 or rtc_config.prefix_attention_schedule is RTCAttentionSchedule.ZEROS:
        return weights

    if rtc_config.prefix_attention_schedule is RTCAttentionSchedule.ONES:
        weights[:prefix_horizon] = 1.0
        return weights

    delay_end = min(inference_delay, prefix_horizon)
    if delay_end > 0:
        weights[:delay_end] = 1.0

    tail_len = prefix_horizon - delay_end
    if tail_len <= 0:
        return weights

    if rtc_config.prefix_attention_schedule is RTCAttentionSchedule.LINEAR:
        weights[delay_end:prefix_horizon] = torch.linspace(1.0, 0.0, tail_len, device=device, dtype=dtype)
    elif rtc_config.prefix_attention_schedule is RTCAttentionSchedule.EXP:
        weights[delay_end:prefix_horizon] = torch.exp(
            -torch.linspace(0.0, 5.0, tail_len, device=device, dtype=dtype)
        )
    else:
        raise ValueError(f"Unsupported RTC prefix attention schedule: {rtc_config.prefix_attention_schedule}")
    return weights


def prepare_prev_chunk_left_over(
    prev_chunk_left_over: Any,
    *,
    batch_size: int,
    action_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Normalize leftover action shape to ``[B, R, D]``."""

    if prev_chunk_left_over is None:
        return None

    if isinstance(prev_chunk_left_over, torch.Tensor):
        leftover = prev_chunk_left_over.to(device=device, dtype=dtype)
    else:
        leftover = torch.as_tensor(prev_chunk_left_over, device=device, dtype=dtype)

    if leftover.ndim == 2:
        leftover = leftover.unsqueeze(0)
    if leftover.ndim != 3:
        raise ValueError(
            "prev_chunk_left_over must have shape [R, D] or [B, R, D]; "
            f"got {tuple(leftover.shape)}."
        )
    if leftover.shape[-1] != action_dim:
        raise ValueError(
            f"prev_chunk_left_over action_dim mismatch: got {leftover.shape[-1]}, expected {action_dim}."
        )
    if leftover.shape[0] == 1 and batch_size > 1:
        leftover = leftover.expand(batch_size, -1, -1)
    elif leftover.shape[0] != batch_size:
        raise ValueError(
            f"prev_chunk_left_over batch mismatch: got {leftover.shape[0]}, expected {batch_size}."
        )
    return leftover


def apply_rtc_guidance(
    actions: torch.Tensor,
    pred_velocity: torch.Tensor,
    *,
    t_cont: float,
    prev_chunk_left_over: torch.Tensor | None,
    inference_delay: int,
    rtc_config: RTCConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply inference-time RTC guidance to a flow velocity prediction."""

    metadata: dict[str, Any] = {
        "applied": False,
        "guidance_loss": None,
        "guided_prefix_len": 0,
    }
    if not rtc_config.enabled or prev_chunk_left_over is None:
        return pred_velocity, metadata

    batch_size, chunk_size, action_dim = actions.shape
    prefix_horizon = resolve_prefix_attention_horizon(rtc_config, chunk_size)
    guided_len = min(prefix_horizon, int(prev_chunk_left_over.shape[1]), chunk_size)
    if guided_len <= 0 or rtc_config.max_guidance_weight == 0:
        return pred_velocity, metadata

    weights = make_prefix_attention_weights(
        chunk_size,
        inference_delay,
        rtc_config,
        device=actions.device,
        dtype=torch.float32,
    )[:guided_len]
    if torch.count_nonzero(weights).item() == 0:
        return pred_velocity, metadata

    pred_clean = actions + (1.0 - float(t_cont)) * pred_velocity
    diff = pred_clean[:, :guided_len, :] - prev_chunk_left_over[:, :guided_len, :]
    weighted_sq = diff.to(torch.float32).pow(2) * weights.view(1, guided_len, 1)
    denom = (weights.sum() * action_dim * batch_size).clamp_min(1.0)
    loss = weighted_sq.sum() / denom
    grad = torch.autograd.grad(loss, actions, retain_graph=False, create_graph=False)[0]
    guided_velocity = pred_velocity - rtc_config.max_guidance_weight * grad.to(pred_velocity.dtype)

    metadata.update(
        {
            "applied": True,
            "guidance_loss": float(loss.detach().cpu()),
            "guided_prefix_len": int(guided_len),
        }
    )
    return guided_velocity, metadata


def build_rtc_metadata(
    *,
    rtc_config: RTCConfig,
    chunk_size: int,
    action_dim: int,
    prev_chunk_left_over: Any = None,
    inference_delay: int = 0,
    applied: bool,
    server_elapsed_ms: float | None = None,
) -> dict[str, Any]:
    leftover_len = 0
    if prev_chunk_left_over is not None:
        leftover_arr = np.asarray(prev_chunk_left_over)
        if leftover_arr.ndim == 2:
            leftover_len = int(leftover_arr.shape[0])
        elif leftover_arr.ndim == 3:
            leftover_len = int(leftover_arr.shape[1])

    metadata = {
        "applied": bool(applied),
        "chunk_size": int(chunk_size),
        "action_dim": int(action_dim),
        "leftover_len": leftover_len,
        "inference_delay": int(inference_delay),
        "execution_horizon": int(rtc_config.execution_horizon),
        "prefix_attention_horizon": resolve_prefix_attention_horizon(rtc_config, chunk_size),
        "prefix_attention_schedule": rtc_config.prefix_attention_schedule.value,
        "recommended_skip_steps": min(max(int(inference_delay), 0), max(int(chunk_size) - 1, 0)),
    }
    if server_elapsed_ms is not None:
        metadata["server_elapsed_ms"] = float(server_elapsed_ms)
    return metadata


class RTCActionQueue:
    """Thread-safe normalized action queue for RTC clients."""

    def __init__(
        self,
        *,
        execution_horizon: int = 8,
        delay_history_size: int = 10,
        default_delay_steps: int = 0,
        underflow_policy: str = "raise",
    ) -> None:
        self.execution_horizon = int(execution_horizon)
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive.")
        self.delay_history_size = int(delay_history_size)
        self.default_delay_steps = int(default_delay_steps)
        self.underflow_policy = underflow_policy
        self._queue: Deque[np.ndarray] = deque()
        self._delay_steps: Deque[int] = deque(maxlen=self.delay_history_size)
        self._last_action: np.ndarray | None = None

        import threading

        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def reset(self, normalized_chunk: np.ndarray) -> None:
        chunk = self._as_chunk(normalized_chunk)
        with self._lock:
            self._queue.clear()
            self._queue.extend(np.asarray(row, dtype=np.float32).copy() for row in chunk)
            self._last_action = None

    def get_left_over(self) -> np.ndarray | None:
        with self._lock:
            if not self._queue:
                return None
            return np.stack(list(self._queue), axis=0).copy()

    def pop(self) -> np.ndarray:
        with self._lock:
            if self._queue:
                action = self._queue.popleft().copy()
                self._last_action = action.copy()
                return action
            if self.underflow_policy == "hold" and self._last_action is not None:
                return self._last_action.copy()
            raise RuntimeError("RTCActionQueue underflow: no normalized action available.")

    def merge(self, normalized_chunk: np.ndarray, skip_steps: int) -> None:
        chunk = self._as_chunk(normalized_chunk)
        if chunk.shape[0] == 0:
            raise ValueError("normalized_chunk must contain at least one action.")
        skip_steps = min(max(int(skip_steps), 0), chunk.shape[0] - 1)
        suffix = chunk[skip_steps:]
        with self._lock:
            self._queue.clear()
            self._queue.extend(np.asarray(row, dtype=np.float32).copy() for row in suffix)

    def should_request(self) -> bool:
        return len(self) <= self.execution_horizon

    def record_delay_steps(self, delay_steps: int) -> None:
        with self._lock:
            self._delay_steps.append(max(int(delay_steps), 0))

    def estimate_inference_delay(self, method: str = "p90") -> int:
        with self._lock:
            if not self._delay_steps:
                return max(self.default_delay_steps, 0)
            values = np.asarray(list(self._delay_steps), dtype=np.float32)
        method = method.lower()
        if method == "max":
            return int(np.ceil(values.max()))
        if method == "p90":
            return int(np.ceil(np.quantile(values, 0.9)))
        if method == "mean":
            return int(np.ceil(values.mean()))
        raise ValueError(f"Unsupported delay estimate method: {method}")

    @staticmethod
    def _as_chunk(normalized_chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(normalized_chunk, dtype=np.float32)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2:
            raise ValueError(f"normalized_chunk must have shape [H, D] or [1, H, D]; got {chunk.shape}.")
        return chunk
