#!/usr/bin/env bash
set -euo pipefail

################################################################################
scene_idx="${scene_idx:-20250818_152739_Q3720_100_130_part01}"
lidar_type="${lidar_type:-lidar}"

config_file="${config_file:-configs/omnire_extended_cam_lidar.yaml}"
dataset_config="${dataset_config:-qcraft/11cams_${lidar_type}}"
extra_config_info="${extra_config_info:-baseline}"

start_timestep="${start_timestep:-0}"
end_timestep="${end_timestep:--1}"

output_root="${output_root:-output}"
project_name="${project_name:-qcraft_${scene_idx}}"
date_str="${date_str:-$(date +%Y%m%d)}"
run_name="${run_name:-${date_str}_${lidar_type}+cam0_1_2_3_5_6_7_9_10_11_12${extra_config_info}}"

group0_gpu="${group0_gpu:-0}"
group1_gpu="${group1_gpu:-1}"
group0_views="${group0_views:-0,2,5,6,7,12}"
group1_views="${group1_views:-1,3,9,10,11}"
################################################################################

export PYTHONPATH=$(pwd)

run_dir="${output_root}/${project_name}/${run_name}"
mkdir -p "${run_dir}"
cp "$0" "${run_dir}/$(basename "$0")"

echo "Run directory: ${run_dir}"
echo "Group 0: GPU ${group0_gpu}, views ${group0_views}"
echo "Group 1: GPU ${group1_gpu}, views ${group1_views}"

CUDA_VISIBLE_DEVICES=${group0_gpu} python tools/train.py \
    --config_file "${config_file}" \
    --output_root "${output_root}" \
    --project "${project_name}" \
    --run_name "${run_name}" \
    --view_parallel \
    --num_view_groups 2 \
    --view_group_id 0 \
    --view_ids "${group0_views}" \
    --gpu_id "${group0_gpu}" \
    dataset="${dataset_config}" \
    data.scene_idx="${scene_idx}" \
    data.start_timestep="${start_timestep}" \
    data.end_timestep="${end_timestep}" \
    view_parallel.enabled=true \
    view_parallel.group_id=0 &

pid0=$!

CUDA_VISIBLE_DEVICES=${group1_gpu} python tools/train.py \
    --config_file "${config_file}" \
    --output_root "${output_root}" \
    --project "${project_name}" \
    --run_name "${run_name}" \
    --view_parallel \
    --num_view_groups 2 \
    --view_group_id 1 \
    --view_ids "${group1_views}" \
    --gpu_id "${group1_gpu}" \
    dataset="${dataset_config}" \
    data.scene_idx="${scene_idx}" \
    data.start_timestep="${start_timestep}" \
    data.end_timestep="${end_timestep}" \
    view_parallel.enabled=true \
    view_parallel.group_id=1 &

pid1=$!

wait "${pid0}"
wait "${pid1}"

echo "View-parallel training finished."
echo "Group 0 checkpoint: ${run_dir}/group0/checkpoint_final.pth"
echo "Group 1 checkpoint: ${run_dir}/group1/checkpoint_final.pth"
