# Dobot Xtrainer in StarVLA

This directory contains the StarVLA-local workflow for Dobot Xtrainer:

- raw HDF5 -> LeRobot conversion
- StarVLA-compatible modality metadata generation
- baseline training config and launch script
- real-robot deployment entrypoints that reuse the Dobot SDK from `openpi05`

It does **not** modify the original converter in `openpi05`. This is a separate StarVLA-ready path.

## Directory layout

```text
examples/DobotXtrainer/
├── README.md
├── data_conversion/
│   ├── convert_xtrainer_to_lerobot.py
│   └── run_convert_xtrainer.sh
├── eval_files/
│   ├── model2dobotxtrainer_interface.py
│   ├── run_dobot_xtrainer_real.py
│   └── run_policy_server.sh
└── train_files/
    ├── run_train_dobot_xtrainer.sh
    └── starvla_qwenoft_dobot_xtrainer.yaml
```

## Data assumptions

The raw Xtrainer episodes are expected to match the original converter layout:

- state: `/observations/qpos`
- action: `/action`
- cameras:
  - `/observations/images/top`
  - `/observations/images/left_wrist`
  - `/observations/images/right_wrist`

The raw 14D order is assumed to be:

```text
[right_waist,
 right_shoulder,
 right_elbow,
 right_forearm_roll,
 right_wrist_angle,
 right_wrist_rotate,
 right_gripper,
 left_waist,
 left_shoulder,
 left_elbow,
 left_forearm_roll,
 left_wrist_angle,
 left_wrist_rotate,
 left_gripper]
```

## What the converter writes

The converter stores `observation.state` and `action` as flat 14D vectors and writes
`meta/modality.json` so StarVLA sees the logical keys below in the **same raw order**:

- `state.right_joints`
- `state.right_gripper`
- `state.left_joints`
- `state.left_gripper`
- `action.right_joints`
- `action.right_gripper`
- `action.left_joints`
- `action.left_gripper`

This avoids an implicit reorder between raw data and the policy action vector.

## 1. Convert raw Xtrainer episodes

Edit the environment variables in the wrapper or override them inline.

```bash
RAW_DIR=/path/to/xtrainer_hdf5 OUTPUT_ROOT_DIR=playground/Datasets/DobotXtrainer DATASET_NAME=dobot_xtrainer_train TASK_DESCRIPTION="pick up the object and place it at the target location" bash examples/DobotXtrainer/data_conversion/run_convert_xtrainer.sh
```

The final dataset path becomes:

```text
playground/Datasets/DobotXtrainer/dobot_xtrainer_train
```

StarVLA expects `meta/modality.json` to exist. The converter writes it automatically.

## 2. Train a baseline Dobot policy

The provided baseline uses `QwenOFT` with 14D absolute joint/gripper actions.

```bash
DATA_ROOT_DIR=playground/Datasets/DobotXtrainer DATA_NAME=dobot_xtrainer_train RUN_ID=dobot_xtrainer_qwenoft bash examples/DobotXtrainer/train_files/run_train_dobot_xtrainer.sh
```

## 3. Key config choices

`examples/DobotXtrainer/train_files/starvla_qwenoft_dobot_xtrainer.yaml`

- `robot_type: dobot_xtrainer`
- `action_mode: abs`
- `action_dim: 14`
- `state_dim: 14`
- 3-camera input:
  - `cam_high`
  - `cam_left_wrist`
  - `cam_right_wrist`

## 4. Deploy on the real Dobot Xtrainer

This repo now provides a StarVLA-side deployment path that keeps the same
server-client split as `examples/Franka`, while reusing the real robot SDK from:

```text
/home/Xtrainer/ziyu/openpi05/openpi-main
```

### 4.1 Start the policy server

Run from the StarVLA repo root:

```bash
your_ckpt=/path/to/your/checkpoint.pt gpu_id=0 port=5694 bash examples/DobotXtrainer/eval_files/run_policy_server.sh
```

### 4.2 Start the real-robot client

In another terminal, also from the StarVLA repo root:

```bash
python examples/DobotXtrainer/eval_files/run_dobot_xtrainer_real.py     --policy_ckpt_path /path/to/your/checkpoint.pt     --task "Transfer the test tube from the right rack to the left rack."     --host 127.0.0.1     --port 5694     --openpi_root /home/Xtrainer/ziyu/openpi05/openpi-main
```

Useful runtime flags:

- `--arms both|left|right`: only actuate the selected arm(s).
- `--debug_step` or `--debug`: require one empty Enter before every low-level action; `q`, `quit`, or `stop` exits before executing the pending step.
- `--actions_per_chunk 8`: execute only the first `N` actions from each predicted chunk.
- `--max_joint_delta_deg 30`: stop if the predicted joint target jumps too far from the current pose.
- `--skip_reset`: skip the built-in neutral reset motion.
- `--no_gripper`: disable gripper control.

Recommended first real-robot smoke test:

```bash
python examples/DobotXtrainer/eval_files/run_dobot_xtrainer_real.py --policy_ckpt_path /path/to/your/checkpoint.pt --task "Transfer the test tube from the right rack to the left rack." --host 127.0.0.1 --port 5694 --openpi_root /home/Xtrainer/ziyu/openpi05/openpi-main --debug_step --actions_per_chunk 1 --max_steps 20
```

### 4.3 Important deployment details

- `RealEnv` reports joint state in `left + right` order, and this deployment path now assumes the checkpoint state/action order is also `left + right`.
- `model2dobotxtrainer_interface.py` no longer performs any left/right arm swap during inference or execution.
- The real robot `ServoJ` API expects joint angles in degrees, while the collected dataset is stored in radians. The deployment client converts predicted joint commands from radians to degrees immediately before calling the SDK.
- Camera request order is fixed to:
  - `cam_high`
  - `cam_left_wrist`
  - `cam_right_wrist`

## Notes

- The converter fixes compressed-image decoding to RGB before writing frames.
- `robot_type=dobot_xtrainer` is registered inside StarVLA and uses a Dobot-specific
  state/action ordering config.
- If your runtime environment does not already include `lerobot`, install that dependency
  before running the converter or training.
