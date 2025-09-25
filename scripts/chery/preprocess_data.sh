#!/bin/bash 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" 
source ${SCRIPT_DIR}/utils.sh

export PYTHONPATH=$(pwd)

target_dir="data/chery/processed" && mkdir -p "${target_dir}"
source /opt/conda/etc/profile.d/conda.sh && conda activate main

gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
  echo "no gpu found"
  exit 1
fi

scene_ids="clip_1746581703000"

CUDA_VISIBLE_DEVICES=${gpu} python datasets/preprocess.py \
    --data_root data/chery/raw/ \
    --target_dir ${target_dir} \
    --dataset chery \
    --split training \
    --scene_ids $scene_ids \
    --workers 64 \
    --process_keys images lidar calib pose dynamic_masks objects lidar_velocities