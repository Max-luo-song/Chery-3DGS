#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/../utils.sh

export PYTHONPATH=$(pwd)

source /opt/conda/etc/profile.d/conda.sh && conda activate main

start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
  echo "no gpu found"
  exit 1
fi

CUDA_VISIBLE_DEVICES=${gpu} python tools/train.py \
    --config_file configs/0901omnire_chery_visual.yaml \
    --output_root output \
    --project omnire \
    --run_name 0902_chery_visual_front_main \
    dataset=chery/1cams_visual \
    data.scene_idx=clip_1746752396800 \
    data.pixel_source.cameras=[0] \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
