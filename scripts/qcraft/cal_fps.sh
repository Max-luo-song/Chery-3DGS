#!/bin/bash
set -euo pipefail

gpu=-1
ckpt_path="${1:-output/qcraft_20250818_152739_Q3720_100_130_part01/20260506_lidar+cam0_1_2_3_5_6_7_9_10_11_12baseline/checkpoint_final.pth}"
num_frames="${2:-0}"
warmup_frames="${3:-20}"
split="${4:-full}"
camera_ids="${5:-}"

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
echo "Benchmark split: ${split}, num_frames: ${num_frames}, warmup_frames: ${warmup_frames}"
if [ -n "${camera_ids}" ]; then
    echo "Benchmark camera ids: ${camera_ids}"
else
    echo "Benchmark camera ids: all"
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate drivestudio

export PYTHONPATH=$(pwd)
cmd=(python tools/cal_fps.py
    --resume_from "${ckpt_path}" \
    --num_frames "${num_frames}" \
    --warmup_frames "${warmup_frames}" \
    --split "${split}")

if [ -n "${camera_ids}" ]; then
    cmd+=(--camera_ids "${camera_ids}")
fi

CUDA_VISIBLE_DEVICES=${gpu} "${cmd[@]}"
