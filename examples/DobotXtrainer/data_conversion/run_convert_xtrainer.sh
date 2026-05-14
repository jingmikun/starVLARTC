#!/usr/bin/env bash
set -euo pipefail

RAW_DIR=${RAW_DIR:-/path/to/xtrainer_hdf5}
OUTPUT_ROOT_DIR=${OUTPUT_ROOT_DIR:-playground/Datasets/DobotXtrainer}
DATASET_NAME=${DATASET_NAME:-dobot_xtrainer_train}
TASK_DESCRIPTION=${TASK_DESCRIPTION:-pick up the object and place it at the target location}
FPS=${FPS:-25}
MIN_FRAMES=${MIN_FRAMES:-100}
MODE=${MODE:-video}
ROBOT_TYPE=${ROBOT_TYPE:-dobot_xtrainer}
VIDEO_BACKEND=${VIDEO_BACKEND:-}

CMD=(
  "python"
  examples/DobotXtrainer/data_conversion/convert_xtrainer_to_lerobot.py
  --raw-dir "${RAW_DIR}"
  --output-root-dir "${OUTPUT_ROOT_DIR}"
  --dataset-name "${DATASET_NAME}"
  --task "${TASK_DESCRIPTION}"
  --fps "${FPS}"
  --min-frames "${MIN_FRAMES}"
  --mode "${MODE}"
  --robot-type "${ROBOT_TYPE}"
)

if [[ -n "${VIDEO_BACKEND}" ]]; then
  CMD+=(--video-backend "${VIDEO_BACKEND}")
fi

printf 'Running conversion command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
