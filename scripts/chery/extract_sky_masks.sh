#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/utils.sh

segformer_path=/opt/SegFormer

source /opt/conda/etc/profile.d/conda.sh && conda activate segformer

gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
  echo "no gpu found"
  exit 1
fi

CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/extract_masks.py \
    --data_root data/chery/processed/training \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --split_file data/chery_example_scenes.txt \
    --process_dynamic_mask
