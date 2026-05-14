from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import time

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.framework.share_tools import read_mode_config


ENV_LEFT_RIGHT_ORDER = np.arange(14, dtype=np.int64)
MODEL_RIGHT_LEFT_ORDER = ENV_LEFT_RIGHT_ORDER
MODEL_GRIPPER_INDICES = (6, 13)


class DobotXtrainerModelClient:
    """Client-side wrapper for StarVLA DobotXtrainer deployment."""

    def __init__(
        self,
        policy_ckpt_path: str,
        host: str = "127.0.0.1",
        port: int = 5694,
        unnorm_key: str | None = None,
        image_size: list[int] | None = None,
        binary_threshold: float = 0.49,
    ) -> None:
        self.policy_ckpt_path = Path(policy_ckpt_path)
        self.host = host
        self.port = port
        self.unnorm_key = unnorm_key
        self.binary_threshold = binary_threshold
        self.client = WebsocketClientPolicy(host=host, port=port)

        self.model_config, self.norm_stats = read_mode_config(self.policy_ckpt_path)
        self.unnorm_key = self._check_unnorm_key(self.norm_stats, self.unnorm_key)
        self.image_size = image_size or self.model_config["datasets"]["vla_data"].get("image_size", [224, 224])
        self.action_mode = self.model_config["datasets"]["vla_data"].get("action_mode", "abs")
        self.action_chunk_size = self.model_config["framework"]["action_model"].get("future_action_window_size", 15) + 1
        self.action_norm_stats = self._get_stats("action")
        self.state_norm_stats = self._get_stats("state")

        self.task_description: str | None = None
        self.initial_state_model: np.ndarray | None = None
        self.prev_action_model: np.ndarray | None = None

    def reset(self, task_description: str | None = None) -> None:
        self.task_description = task_description
        self.initial_state_model = None
        self.prev_action_model = None

    def predict_action_chunk(
        self,
        images: list[np.ndarray],
        task_description: str,
        qpos_left_right: np.ndarray,
        use_ddim: bool = False,
        num_ddim_steps: int = 4,
    ) -> np.ndarray:
        normalized_actions, _ = self.predict_normalized_action_chunk(
            images=images,
            task_description=task_description,
            qpos_left_right=qpos_left_right,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
        )
        raw_state_model = self.reorder_env_state_to_model(qpos_left_right)
        return self.normalized_action_to_model_action(normalized_actions, raw_state_model)

    def predict_normalized_action_chunk(
        self,
        images: list[np.ndarray],
        task_description: str,
        qpos_left_right: np.ndarray,
        *,
        use_rtc: bool = False,
        prev_chunk_left_over: np.ndarray | None = None,
        inference_delay: int = 0,
        execution_horizon: int = 8,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
        gripper_guidance_scale: float = 0.0,
        use_ddim: bool = False,
        num_ddim_steps: int = 4,
    ) -> tuple[np.ndarray, dict]:
        """
        Return normalized_actions [H,D] and metadata dict.
        Does not unnormalize.
        """
        if task_description != self.task_description:
            self.reset(task_description)

        raw_state_model = self.reorder_env_state_to_model(qpos_left_right)
        if self.initial_state_model is None:
            self.initial_state_model = raw_state_model.copy()

        normalized_state = self.normalize_state(raw_state_model).reshape(1, -1).astype(np.float32)
        resized_images = [self._resize_image(image) for image in images]
        example = {
            "image": resized_images,
            "lang": task_description,
            "state": normalized_state,
        }

        if use_rtc:
            request = {
                "type": "infer_rtc",
                "request_id": f"dobotxtrainer-rtc-{time.time_ns()}",
                "payload": {
                    "examples": [example],
                    "rtc": {
                        "enabled": True,
                        "prev_chunk_left_over": None
                        if prev_chunk_left_over is None
                        else np.asarray(prev_chunk_left_over, dtype=np.float32),
                        "inference_delay": int(inference_delay),
                        "execution_horizon": int(execution_horizon),
                        "prefix_attention_schedule": prefix_attention_schedule,
                        "max_guidance_weight": float(max_guidance_weight),
                        "gripper_guidance_scale": float(gripper_guidance_scale),
                        "debug": False,
                    },
                },
            }
        else:
            request = {
                "examples": [example],
                "do_sample": False,
                "use_ddim": use_ddim,
                "num_ddim_steps": num_ddim_steps,
            }

        response = self.client.predict_action(request)
        normalized_actions = self.parse_response(response)
        metadata = response.get("data", response).get("rtc", {})
        return normalized_actions.astype(np.float32, copy=False), metadata

    def normalized_action_to_model_action(
        self,
        normalized_actions: np.ndarray,
        raw_state_model: np.ndarray,
    ) -> np.ndarray:
        """
        Unnormalize actions and apply abs/delta/rel conversion.
        """
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        was_1d = normalized_actions.ndim == 1
        if was_1d:
            normalized_actions = normalized_actions.reshape(1, -1)
        if normalized_actions.ndim != 2:
            raise ValueError(f"Expected normalized action shape [H,D] or [D], got {normalized_actions.shape}")

        model_actions = self.unnormalize_actions(normalized_actions)
        if self.action_mode == "delta":
            model_actions = self._delta_to_absolute(model_actions, raw_state_model)
        elif self.action_mode == "rel":
            model_actions = self._rel_to_absolute(model_actions)
        elif self.action_mode != "abs":
            raise ValueError(f"Unsupported action mode: {self.action_mode}")

        return model_actions[0] if was_1d else model_actions

    def commit_executed_action(self, executed_action_model: np.ndarray) -> None:
        if self.action_mode == "delta":
            self.prev_action_model = np.array(executed_action_model, dtype=np.float32).copy()

    def reorder_env_state_to_model(self, qpos_left_right: np.ndarray) -> np.ndarray:
        qpos_left_right = np.asarray(qpos_left_right, dtype=np.float32).reshape(-1)
        if qpos_left_right.shape[0] != 14:
            raise ValueError(f"Expected 14D Dobot qpos, got shape {qpos_left_right.shape}")
        return qpos_left_right[ENV_LEFT_RIGHT_ORDER]

    def reorder_model_action_to_env(self, action_right_left: np.ndarray) -> np.ndarray:
        action_right_left = np.asarray(action_right_left, dtype=np.float32).reshape(-1)
        if action_right_left.shape[0] != 14:
            raise ValueError(f"Expected 14D Dobot action, got shape {action_right_left.shape}")
        return action_right_left[MODEL_RIGHT_LEFT_ORDER]

    def model_action_to_robot_command(self, action_right_left: np.ndarray) -> np.ndarray:
        action_left_right = self.reorder_model_action_to_env(action_right_left).astype(np.float32, copy=True)
        action_left_right[:6] = np.rad2deg(action_left_right[:6])
        action_left_right[7:13] = np.rad2deg(action_left_right[7:13])
        action_left_right[6] = float(np.clip(action_left_right[6], 0.0, 1.0))
        action_left_right[13] = float(np.clip(action_left_right[13], 0.0, 1.0))
        return action_left_right

    def normalize_state(self, raw_state_model: np.ndarray) -> np.ndarray:
        stats = self.state_norm_stats
        raw_state_model = np.asarray(raw_state_model, dtype=np.float32)
        if stats is None:
            return raw_state_model

        if self._is_min_max_stats(stats):
            low = np.asarray(stats["min"], dtype=np.float32)
            high = np.asarray(stats["max"], dtype=np.float32)
        elif self._is_q99_stats(stats):
            low = np.asarray(stats["q01"], dtype=np.float32)
            high = np.asarray(stats["q99"], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported state statistics keys: {list(stats.keys())}")

        denom = high - low
        mask = denom != 0
        normalized = np.zeros_like(raw_state_model, dtype=np.float32)
        normalized[mask] = 2.0 * (raw_state_model[mask] - low[mask]) / denom[mask] - 1.0
        normalized[~mask] = 0.0
        normalized = np.clip(normalized, -1.0, 1.0)
        normalized[list(MODEL_GRIPPER_INDICES)] = (raw_state_model[list(MODEL_GRIPPER_INDICES)] > self.binary_threshold).astype(
            np.float32
        )
        return normalized

    def unnormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        stats = self.action_norm_stats
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
        normalized_actions[:, list(MODEL_GRIPPER_INDICES)] = (
            normalized_actions[:, list(MODEL_GRIPPER_INDICES)] > self.binary_threshold
        ).astype(np.float32)
        mask = np.asarray(stats.get("mask", np.ones(normalized_actions.shape[-1], dtype=bool)), dtype=bool)

        if self._is_min_max_stats(stats):
            low = np.asarray(stats["min"], dtype=np.float32)
            high = np.asarray(stats["max"], dtype=np.float32)
            return np.where(mask, 0.5 * (normalized_actions + 1.0) * (high - low) + low, normalized_actions)

        if self._is_q99_stats(stats):
            low = np.asarray(stats["q01"], dtype=np.float32)
            high = np.asarray(stats["q99"], dtype=np.float32)
            return np.where(mask, 0.5 * (normalized_actions + 1.0) * (high - low) + low, normalized_actions)

        raise ValueError(f"Unsupported action statistics keys: {list(stats.keys())}")

    def parse_response(self, result: dict[str, Any]) -> np.ndarray:
        data = result.get("data", result)
        for key in ("normalized_actions", "actions", "action"):
            if key not in data:
                continue
            actions = np.asarray(data[key], dtype=np.float32)
            if actions.ndim == 3:
                return actions[0]
            if actions.ndim == 2:
                return actions
            if actions.ndim == 1:
                return actions.reshape(1, -1)
        raise KeyError(f"Could not extract action chunk from response keys: {list(data.keys())}")

    def get_current_state_delta_deg(
        self,
        qpos_left_right: np.ndarray,
        command_left_right_deg: np.ndarray,
        arms: str = "both",
    ) -> float:
        qpos_left_right = np.asarray(qpos_left_right, dtype=np.float32).reshape(-1)
        current_deg = qpos_left_right.copy()
        current_deg[:6] = np.rad2deg(current_deg[:6])
        current_deg[7:13] = np.rad2deg(current_deg[7:13])

        if arms == "left":
            joint_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        elif arms == "right":
            joint_indices = np.array([7, 8, 9, 10, 11, 12], dtype=np.int64)
        else:
            joint_indices = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
        return float(np.max(np.abs(command_left_right_deg[joint_indices] - current_deg[joint_indices])))

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state_model: np.ndarray) -> np.ndarray:
        abs_actions = np.zeros_like(delta_actions)
        base = self.prev_action_model if self.prev_action_model is not None else current_state_model
        for idx in range(len(delta_actions)):
            abs_actions[idx] = delta_actions[idx] + base
            abs_actions[idx, list(MODEL_GRIPPER_INDICES)] = delta_actions[idx, list(MODEL_GRIPPER_INDICES)]
            base = abs_actions[idx]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        if self.initial_state_model is None:
            raise RuntimeError("Relative action mode requires an initialized state.")
        abs_actions = np.zeros_like(rel_actions)
        for idx in range(len(rel_actions)):
            abs_actions[idx] = rel_actions[idx] + self.initial_state_model
            abs_actions[idx, list(MODEL_GRIPPER_INDICES)] = rel_actions[idx, list(MODEL_GRIPPER_INDICES)]
        return abs_actions

    def _get_stats(self, stat_name: str) -> dict[str, Any] | None:
        stats_root = self.norm_stats[self.unnorm_key]
        if stat_name in stats_root:
            return stats_root[stat_name]
        if self.action_mode in stats_root and stat_name in stats_root[self.action_mode]:
            return stats_root[self.action_mode][stat_name]
        return None

    @staticmethod
    def load_dataset_statistics(policy_ckpt_path: str, output_path: str | None = None) -> dict[str, Any]:
        _, norm_stats = read_mode_config(Path(policy_ckpt_path))
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(norm_stats, file, indent=2, ensure_ascii=False)
        return norm_stats

    @staticmethod
    def _check_unnorm_key(norm_stats: dict[str, Any], unnorm_key: str | None) -> str:
        if unnorm_key is None:
            return next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            raise KeyError(f"Unknown unnorm_key={unnorm_key}. Available keys: {list(norm_stats.keys())}")
        return unnorm_key

    @staticmethod
    def _is_min_max_stats(stats: dict[str, Any]) -> bool:
        return stats is not None and "min" in stats and "max" in stats

    @staticmethod
    def _is_q99_stats(stats: dict[str, Any]) -> bool:
        return stats is not None and "q01" in stats and "q99" in stats

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        height, width = int(self.image_size[0]), int(self.image_size[1])
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
