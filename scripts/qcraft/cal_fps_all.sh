#!/bin/bash
set -euo pipefail

gpu=-1
ckpt_path="${1:-output/qcraft_20250818_152739_Q3720_100_130_part01/20260506_lidar+cam0_1_2_3_5_6_7_9_10_11_12baseline/checkpoint_final.pth}"
num_frames="${2:-0}"
warmup_frames="${3:-20}"
split="${4:-full}"

source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
        exit 1
    fi
fi

echo "Using GPU: ${gpu}"
echo "Using checkpoint: ${ckpt_path}"
echo "Benchmark all cameras separately, split: ${split}, num_frames: ${num_frames}, warmup_frames: ${warmup_frames}"

source /opt/conda/etc/profile.d/conda.sh
conda activate drivestudio

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=${gpu} python tools/cal_fps.py \
    --resume_from "${ckpt_path}" \
    --num_frames "${num_frames}" \
    --warmup_frames "${warmup_frames}" \
    --split "${split}" \
    --all_cameras
