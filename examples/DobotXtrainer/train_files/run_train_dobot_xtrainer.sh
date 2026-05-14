#!/usr/bin/env bash
set -euo pipefail

export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1000}
export WANDB_MODE=offline

PYTHON_BIN=${PYTHON_BIN:-accelerate}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}
NUM_PROCESSES=${NUM_PROCESSES:-2}
FRAMEWORK_NAME=${FRAMEWORK_NAME:-QwenGR00T}
BASE_VLM=${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}
CONFIG_YAML=${CONFIG_YAML:-./examples/DobotXtrainer/train_files/starvla_qwengr00t_dobot_xtrainer.yaml}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-playground/Datasets/DobotXtrainer}
DATA_NAME=${DATA_NAME:-task_00031_ext}
ROBOT_TYPE=${ROBOT_TYPE:-dobot_xtrainer}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-./results/PostCheckpoints/DobotXtrainer}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
RUN_ID="0413_tube-xtrainer_${FRAMEWORK_NAME}_bz${PER_DEVICE_BATCH_SIZE}"
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-100000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
WANDB_PROJECT=${WANDB_PROJECT:-starvla_dobot_xtrainer}
WANDB_ENTITY=${WANDB_ENTITY:-rhos-ziyu}

OUTPUT_DIR=${RUN_ROOT_DIR}/${RUN_ID}
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

"${PYTHON_BIN}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR}" \
  --datasets.vla_data.data_name "${DATA_NAME}" \
  --datasets.vla_data.robot_type "${ROBOT_TYPE}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --datasets.vla_data.video_backend torchvision_av
