"""Convert Dobot Xtrainer HDF5 episodes into a StarVLA-ready LeRobot dataset.

This converter keeps the raw state/action storage as flat 14D vectors and writes a
StarVLA-compatible ``meta/modality.json`` that exposes the Dobot-specific logical
keys in raw order:

- right_joints  [0:6]
- right_gripper [6:7]
- left_joints   [7:13]
- left_gripper  [13:14]

The generated dataset can be consumed directly by StarVLA with
``robot_type=dobot_xtrainer``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import tqdm
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

CAMERA_KEY_MAP = {
    "top": "cam_high",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}

RAW_MOTOR_ORDER = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
]

DOBOT_MODALITY_METADATA = {
    "action": {
        "right_joints": {"start": 0, "end": 6, "original_key": "action"},
        "right_gripper": {"start": 6, "end": 7, "original_key": "action"},
        "left_joints": {"start": 7, "end": 13, "original_key": "action"},
        "left_gripper": {"start": 13, "end": 14, "original_key": "action"},
    },
    "state": {
        "right_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
        "right_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
        "left_joints": {"start": 7, "end": 13, "original_key": "observation.state"},
        "left_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
    },
    "video": {
        "cam_high": {"original_key": "observation.images.cam_high"},
        "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
        "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


@dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    dataset_name: str,
    robot_type: str,
    mode: str,
    fps: int,
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(RAW_MOTOR_ORDER),),
            "names": [RAW_MOTOR_ORDER],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(RAW_MOTOR_ORDER),),
            "names": [RAW_MOTOR_ORDER],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(RAW_MOTOR_ORDER),),
            "names": [RAW_MOTOR_ORDER],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(RAW_MOTOR_ORDER),),
            "names": [RAW_MOTOR_ORDER],
        }

    for target_camera in CAMERA_KEY_MAP.values():
        features[f"observation.images.{target_camera}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    temporary_dataset_path = HF_LEROBOT_HOME / dataset_name
    if temporary_dataset_path.exists():
        shutil.rmtree(temporary_dataset_path)

    return LeRobotDataset.create(
        repo_id=dataset_name,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def _load_images_per_camera(ep: h5py.File) -> dict[str, np.ndarray]:
    images_by_camera: dict[str, np.ndarray] = {}
    for source_camera, target_camera in CAMERA_KEY_MAP.items():
        source_key = f"/observations/images/{source_camera}"
        if source_key not in ep:
            raise KeyError(f"Missing camera stream: {source_key}")

        image_dataset = ep[source_key]
        if image_dataset.ndim == 4:
            image_array = image_dataset[:]
        else:
            import cv2

            decoded_frames = []
            for encoded_image in image_dataset:
                encoded_np = np.frombuffer(encoded_image, np.uint8)
                frame_bgr = cv2.imdecode(encoded_np, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise ValueError(f"Failed to decode frame from {source_key}")
                decoded_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            image_array = np.asarray(decoded_frames)

        images_by_camera[target_camera] = image_array

    return images_by_camera


def load_raw_episode_data(
    episode_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(episode_path, "r") as episode_file:
        state = torch.from_numpy(episode_file["/observations/qpos"][:]).float()
        action = torch.from_numpy(episode_file["/action"][:]).float()

        velocity = None
        if "/observations/qvel" in episode_file:
            velocity = torch.from_numpy(episode_file["/observations/qvel"][:]).float()

        effort = None
        if "/observations/effort" in episode_file:
            effort = torch.from_numpy(episode_file["/observations/effort"][:]).float()

        images_per_camera = _load_images_per_camera(episode_file)

    return images_per_camera, state, action, velocity, effort


def _iter_hdf5_files(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.iterdir() if path.suffix == ".hdf5")


def _write_modality_json(dataset_path: Path) -> None:
    modality_path = dataset_path / "meta" / "modality.json"
    modality_path.parent.mkdir(parents=True, exist_ok=True)
    with open(modality_path, "w", encoding="utf-8") as file:
        json.dump(DOBOT_MODALITY_METADATA, file, indent=4, ensure_ascii=False)


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: Iterable[Path],
    task: str,
    *,
    min_frames: int,
) -> None:
    print(f"Populating dataset with task description: {task}")

    for episode_index, episode_path in enumerate(tqdm.tqdm(list(hdf5_files))):
        images_per_camera, state, action, velocity, effort = load_raw_episode_data(episode_path)
        frame_count = int(state.shape[0])
        if frame_count < min_frames:
            print(
                f"Skipping episode {episode_index} ({episode_path.name}): "
                f"{frame_count} frames < min_frames={min_frames}"
            )
            continue

        for frame_index in range(frame_count):
            frame = {
                "observation.state": state[frame_index],
                "action": action[frame_index],
            }
            for camera_name, image_array in images_per_camera.items():
                frame[f"observation.images.{camera_name}"] = image_array[frame_index]
            if velocity is not None:
                frame["observation.velocity"] = velocity[frame_index]
            if effort is not None:
                frame["observation.effort"] = effort[frame_index]
            dataset.add_frame(frame, task=task)

        dataset.save_episode()


def convert_xtrainer_dataset(
    raw_dir: Path,
    output_root_dir: Path,
    dataset_name: str,
    task: str,
    *,
    fps: int,
    min_frames: int,
    mode: str,
    robot_type: str,
    dataset_config: DatasetConfig,
) -> Path:
    raw_dir = raw_dir.expanduser().resolve()
    output_root_dir = output_root_dir.expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    hdf5_files = _iter_hdf5_files(raw_dir)
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {raw_dir}")

    dataset = create_empty_dataset(
        dataset_name=dataset_name,
        robot_type=robot_type,
        mode=mode,
        fps=fps,
        has_velocity=has_velocity(hdf5_files[0]),
        has_effort=has_effort(hdf5_files[0]),
        dataset_config=dataset_config,
    )
    populate_dataset(dataset, hdf5_files, task, min_frames=min_frames)

    temporary_dataset_path = HF_LEROBOT_HOME / dataset_name
    final_dataset_path = output_root_dir / dataset_name
    if final_dataset_path.exists():
        shutil.rmtree(final_dataset_path)
    final_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if temporary_dataset_path.resolve() != final_dataset_path.resolve():
        shutil.move(str(temporary_dataset_path), str(final_dataset_path))
    _write_modality_json(final_dataset_path)
    print(f"Wrote StarVLA-ready dataset to: {final_dataset_path}")
    return final_dataset_path


def has_velocity(first_episode_path: Path) -> bool:
    with h5py.File(first_episode_path, "r") as episode_file:
        return "/observations/qvel" in episode_file


def has_effort(first_episode_path: Path) -> bool:
    with h5py.File(first_episode_path, "r") as episode_file:
        return "/observations/effort" in episode_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing Xtrainer .hdf5 episodes")
    parser.add_argument(
        "--output-root-dir",
        type=Path,
        required=True,
        help="Destination root directory; final dataset path becomes <output-root-dir>/<dataset-name>",
    )
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset folder name used by StarVLA")
    parser.add_argument("--task", type=str, required=True, help="Task description stored in the LeRobot dataset")
    parser.add_argument("--fps", type=int, default=25, help="FPS written into LeRobot metadata")
    parser.add_argument("--min-frames", type=int, default=100, help="Skip episodes shorter than this frame count")
    parser.add_argument(
        "--mode",
        choices=["video", "image"],
        default="video",
        help="How image observations are stored in the LeRobot dataset",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="dobot_xtrainer",
        help="Robot type written into LeRobot metadata and used by StarVLA training",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default=None,
        help="Optional LeRobot video backend override",
    )
    parser.add_argument(
        "--use-images",
        action="store_true",
        help="Store frames as images instead of videos (overrides --mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = DatasetConfig(
        use_videos=not args.use_images,
        video_backend=args.video_backend,
    )
    mode = "image" if args.use_images else args.mode
    convert_xtrainer_dataset(
        raw_dir=args.raw_dir,
        output_root_dir=args.output_root_dir,
        dataset_name=args.dataset_name,
        task=args.task,
        fps=args.fps,
        min_frames=args.min_frames,
        mode=mode,
        robot_type=args.robot_type,
        dataset_config=dataset_config,
    )


if __name__ == "__main__":
    main()
