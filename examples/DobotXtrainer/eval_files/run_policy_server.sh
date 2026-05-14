#!/bin/bash
set -euo pipefail

# export PYTHONPATH=$(pwd):${PYTHONPATH}

star_vla_python=${star_vla_python:-python}
your_ckpt=${your_ckpt:-results/Checkpoints/dobot_xtrainer_qwenoft/checkpoints/steps_50000_pytorch_model.pt}
gpu_id=${gpu_id:-0}
port=${port:-5694}

CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py     --ckpt_path "${your_ckpt}"     --port "${port}"     --use_bf16
