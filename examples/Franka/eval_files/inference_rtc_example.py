"""Franka RTC inference template.

This file shows the complete real-time chunking loop for Franka-like robots.
It is intentionally a template: replace ``YourRobotEnv`` with the real robot
environment and keep the normalized queue semantics unchanged.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.Franka.eval_files.rtc_action_queue import RTCActionQueue


def load_action_norm_stats(json_path: str, embodiment_key: str = "franka") -> Dict[str, np.ndarray]:
    with open(json_path, "r") as f:
        stats_data = json.load(f)

    if embodiment_key in stats_data:
        stats_data = stats_data[embodiment_key]
    if "action" in stats_data:
        stats_data = stats_data["action"]

    norm_stats = {
        "min": np.asarray(stats_data.get("min", stats_data.get("low", [])), dtype=np.float32),
        "max": np.asarray(stats_data.get("max", stats_data.get("high", [])), dtype=np.float32),
    }
    if "mask" in stats_data:
        norm_stats["mask"] = np.asarray(stats_data["mask"], dtype=bool)
    return norm_stats


def unnormalize_action(normalized_action: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert one normalized action to the robot action space."""

    action = np.asarray(normalized_action, dtype=np.float32).copy()
    if action.ndim != 1:
        raise ValueError(f"Expected one action shaped [D], got {action.shape}.")

    action = np.clip(action, -1.0, 1.0)
    action_dim = action.shape[-1]
    if action_dim == 14:
        action[6] = -1.0 if action[6] < 0.5 else 1.0
        action[13] = -1.0 if action[13] < 0.5 else 1.0
    elif action_dim >= 7:
        action[6] = -1.0 if action[6] < 0.5 else 1.0

    action_low = np.asarray(action_norm_stats["min"], dtype=np.float32)
    action_high = np.asarray(action_norm_stats["max"], dtype=np.float32)
    mask = np.asarray(action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool)), dtype=bool)
    return np.where(mask, 0.5 * (action + 1.0) * (action_high - action_low) + action_low, action)


def build_examples(obs: dict, task_instruction: str) -> list[dict]:
    example = {
        "image": obs["images"],
        "lang": task_instruction,
    }
    if "state" in obs and obs["state"] is not None:
        example["state"] = obs["state"]
    return [example]


def parse_rtc_response(result: dict) -> Tuple[np.ndarray, dict]:
    if not result.get("ok", result.get("status") == "ok"):
        raise RuntimeError(f"RTC inference failed: {result.get('error', result)}")

    data = result.get("data", result)
    normalized_actions = np.asarray(data["normalized_actions"], dtype=np.float32)
    if normalized_actions.ndim == 3:
        normalized_actions = normalized_actions[0]
    if normalized_actions.ndim != 2:
        raise ValueError(f"Expected normalized_actions [H, D] or [B, H, D], got {normalized_actions.shape}.")
    return normalized_actions, data.get("rtc", {})


class YourRobotEnv:
    def reset(self) -> dict:
        raise NotImplementedError

    def get_obs(self) -> dict:
        raise NotImplementedError

    def step(self, action: np.ndarray):
        raise NotImplementedError

    def safety_check_action(self, action: np.ndarray) -> np.ndarray:
        return action


class FrankaRTCRunner:
    def __init__(
        self,
        *,
        client: WebsocketClientPolicy,
        env: YourRobotEnv,
        task_instruction: str,
        action_norm_stats: Dict[str, np.ndarray],
        control_hz: float,
        execution_horizon: int = 8,
        max_guidance_weight: float = 5.0,
        prefix_attention_schedule: str = "EXP",
        delay_estimate_method: str = "p90",
    ) -> None:
        self.client = client
        self.env = env
        self.task_instruction = task_instruction
        self.action_norm_stats = action_norm_stats
        self.control_hz = float(control_hz)
        self.execution_horizon = int(execution_horizon)
        self.max_guidance_weight = float(max_guidance_weight)
        self.prefix_attention_schedule = prefix_attention_schedule
        self.delay_estimate_method = delay_estimate_method
        self.queue = RTCActionQueue(execution_horizon=self.execution_horizon, underflow_policy="raise")
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending: Future | None = None

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def warm_start(self, obs: dict) -> dict:
        rtc_payload = self._rtc_payload(prev_chunk_left_over=None)
        result = self.client.predict_action_rtc(
            build_examples(obs, self.task_instruction),
            rtc=rtc_payload,
            request_id="rtc-warm-start",
        )
        normalized_chunk, metadata = parse_rtc_response(result)
        self.queue.reset(normalized_chunk)
        return metadata

    def run_episode(self, max_steps: int = 500) -> None:
        obs = self.env.reset()
        warm_meta = self.warm_start(obs)
        print(f"[rtc] warm_start queue_len={len(self.queue)} meta={warm_meta}")

        done = False
        for step_idx in range(max_steps):
            tick_start = time.perf_counter()
            self._merge_pending_if_ready()
            if self.pending is None and self.queue.should_request():
                self._start_async_request(obs, step_idx)

            normalized_action = self.queue.pop()
            action = unnormalize_action(normalized_action, self.action_norm_stats)
            action = self.env.safety_check_action(action)
            obs, _, done, truncated, info = self.env.step(action)
            done = bool(done or truncated)

            if done:
                break

            elapsed = time.perf_counter() - tick_start
            sleep_s = max((1.0 / self.control_hz) - elapsed, 0.0)
            if sleep_s > 0:
                time.sleep(sleep_s)

        self._merge_pending_if_ready(block=True)

    def _rtc_payload(self, prev_chunk_left_over: np.ndarray | None) -> dict:
        return {
            "enabled": True,
            "prev_chunk_left_over": prev_chunk_left_over,
            "inference_delay": self.queue.estimate_inference_delay(self.delay_estimate_method),
            "execution_horizon": self.execution_horizon,
            "max_guidance_weight": self.max_guidance_weight,
            "prefix_attention_schedule": self.prefix_attention_schedule,
            "debug": False,
        }

    def _start_async_request(self, obs: dict, step_idx: int) -> None:
        request_obs = dict(obs)
        left_over = self.queue.get_left_over()
        rtc_payload = self._rtc_payload(left_over)
        request_id = f"rtc-step-{step_idx}"
        self.pending = self.executor.submit(self._request_chunk, request_obs, rtc_payload, request_id)

    def _request_chunk(self, obs: dict, rtc_payload: dict, request_id: str) -> tuple[np.ndarray, int, dict]:
        request_t = time.perf_counter()
        result = self.client.predict_action_rtc(
            build_examples(obs, self.task_instruction),
            rtc=rtc_payload,
            request_id=request_id,
        )
        response_t = time.perf_counter()
        normalized_chunk, metadata = parse_rtc_response(result)
        actual_delay_steps = int(round((response_t - request_t) * self.control_hz))
        skip_steps = min(max(actual_delay_steps, 0), normalized_chunk.shape[0] - 1)
        metadata = dict(metadata)
        metadata["actual_delay_steps"] = actual_delay_steps
        metadata["actual_skip_steps"] = skip_steps
        return normalized_chunk, skip_steps, metadata

    def _merge_pending_if_ready(self, block: bool = False) -> None:
        if self.pending is None:
            return
        if not block and not self.pending.done():
            return
        normalized_chunk, skip_steps, metadata = self.pending.result()
        self.queue.record_delay_steps(metadata["actual_delay_steps"])
        self.queue.merge(normalized_chunk, skip_steps=skip_steps)
        self.pending = None
        print(
            "[rtc] "
            f"queue_len={len(self.queue)} "
            f"estimated_delay={self.queue.estimate_inference_delay(self.delay_estimate_method)} "
            f"actual_delay={metadata['actual_delay_steps']} "
            f"skip_steps={skip_steps} "
            f"server_elapsed_ms={metadata.get('server_elapsed_ms')}"
        )


def main() -> None:
    policy_host = "127.0.0.1"
    policy_port = 5694
    task_instruction = "Pick up the pink cube and place it into the black box."
    action_stats_path = "/path/to/dataset_statistics.json"
    embodiment_key = "franka"
    control_hz = 10.0

    action_norm_stats = load_action_norm_stats(action_stats_path, embodiment_key=embodiment_key)
    client = WebsocketClientPolicy(host=policy_host, port=policy_port)
    env = YourRobotEnv()
    runner = FrankaRTCRunner(
        client=client,
        env=env,
        task_instruction=task_instruction,
        action_norm_stats=action_norm_stats,
        control_hz=control_hz,
    )
    try:
        runner.run_episode(max_steps=500)
    finally:
        runner.close()
        client.close()


if __name__ == "__main__":
    main()
