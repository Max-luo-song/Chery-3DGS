#!/usr/bin/env bash
set -euo pipefail

################################################################################
scene_idx="${scene_idx:-20250818_152739_Q3720_100_130_part01}"

lidar_type="${lidar_type:-lidar}"  # lidar（运动补偿前）/visual（纯视觉）
config_file="${config_file:-configs/omnire_extended_cam_lidar.yaml}"
dataset_config="${dataset_config:-qcraft/11cams_${lidar_type}}"
extra_config_info="${extra_config_info:-015912_23610}"

start_timestep="${start_timestep:-0}"
end_timestep="${end_timestep:--1}"

output_root="${output_root:-output}"
project_name="${project_name:-qcraft_${scene_idx}}"
date_str="${date_str:-$(date +%Y%m%d)}"

group0_gpu="${group0_gpu:-0}"
group1_gpu="${group1_gpu:-1}"
train_group0_views="${train_group0_views:-2,3,6,10}"
train_group1_views="${train_group1_views:-0,1,5,9,12}"
eval_group0_views="${eval_group0_views:-2,3,6,7,10}"
eval_group1_views="${eval_group1_views:-0,1,5,9,11,12}"
eval_downscale_when_loading="${eval_downscale_when_loading:-[1]}"
train_view_tag="${train_view_tag:-${train_group0_views//,/_}_${train_group1_views//,/_}}"
run_name="${run_name:-${date_str}_${lidar_type}+traincam${train_view_tag}${extra_config_info}}"
view_order="${view_order:-0,1,2,3,5,6,7,9,10,11,12}"
fps="${fps:-10}"

cleanup_group_eval="${cleanup_group_eval:-1}"
################################################################################

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${repo_root}/.torch_extensions}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

run_dir="${output_root}/${project_name}/${run_name}"

if [ -d "$run_dir" ]; then
    echo "错误：目录 $run_dir 已存在！"
    rm -r "$run_dir"
fi
mkdir -p "$run_dir"

script_name=$(basename "$0")
cp "$0" "${run_dir}/${script_name}"

echo "Run directory: ${run_dir}"
echo "Training group 0: GPU ${group0_gpu}, views ${train_group0_views}"
echo "Training group 1: GPU ${group1_gpu}, views ${train_group1_views}"
echo "Evaluation group 0 views: ${eval_group0_views}"
echo "Evaluation group 1 views: ${eval_group1_views}"

echo "Starting view-parallel training..."
echo "Preparing gsplat CUDA extension cache: ${TORCH_EXTENSIONS_DIR}"
CUDA_VISIBLE_DEVICES=${group0_gpu} python -c "from gsplat.cuda._backend import _C; assert _C is not None"

group0_log="${run_dir}/group0_train.log"
group1_log="${run_dir}/group1_train.log"
echo "Group 0 log: ${group0_log}"
echo "Group 1 log: ${group1_log}"

CUDA_VISIBLE_DEVICES=${group0_gpu} python -u tools/train.py \
    --config_file "${config_file}" \
    --output_root "${output_root}" \
    --project "${project_name}" \
    --run_name "${run_name}" \
    --view_parallel \
    --num_view_groups 2 \
    --view_group_id 0 \
    --view_ids "${train_group0_views}" \
    --gpu_id "${group0_gpu}" \
    dataset="${dataset_config}" \
    data.scene_idx="${scene_idx}" \
    data.start_timestep="${start_timestep}" \
    data.end_timestep="${end_timestep}" \
    view_parallel.enabled=true \
    view_parallel.group_id=0 > "${group0_log}" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=${group1_gpu} python -u tools/train.py \
    --config_file "${config_file}" \
    --output_root "${output_root}" \
    --project "${project_name}" \
    --run_name "${run_name}" \
    --view_parallel \
    --num_view_groups 2 \
    --view_group_id 1 \
    --view_ids "${train_group1_views}" \
    --gpu_id "${group1_gpu}" \
    dataset="${dataset_config}" \
    data.scene_idx="${scene_idx}" \
    data.start_timestep="${start_timestep}" \
    data.end_timestep="${end_timestep}" \
    view_parallel.enabled=true \
    view_parallel.group_id=1 > "${group1_log}" 2>&1 &
pid1=$!

echo "Started group 0 PID ${pid0} on GPU ${group0_gpu}"
echo "Started group 1 PID ${pid1} on GPU ${group1_gpu}"

status0=0
status1=0
wait "${pid0}" || status0=$?
wait "${pid1}" || status1=$?
if [ "${status0}" -ne 0 ] || [ "${status1}" -ne 0 ]; then
    echo "Training failed: group0 status=${status0}, group1 status=${status1}"
    echo "Check logs:"
    echo "  ${group0_log}"
    echo "  ${group1_log}"
    echo "===== group0 log tail ====="
    tail -n 80 "${group0_log}" || true
    echo "===== group1 log tail ====="
    tail -n 80 "${group1_log}" || true
    exit 1
fi

group0_ckpt="${run_dir}/group0/checkpoint_final.pth"
group1_ckpt="${run_dir}/group1/checkpoint_final.pth"

if [ ! -f "${group0_ckpt}" ]; then
    echo "Missing checkpoint: ${group0_ckpt}"
    exit 1
fi
if [ ! -f "${group1_ckpt}" ]; then
    echo "Missing checkpoint: ${group1_ckpt}"
    exit 1
fi

echo "Starting view-parallel evaluation..."
CUDA_VISIBLE_DEVICES=${group0_gpu} python tools/eval.py \
    --resume_from "${group0_ckpt}" \
    --view_parallel \
    --num_view_groups 2 \
    --view_group_id 0 \
    --view_ids "${eval_group0_views}" \
    --gpu_id "${group0_gpu}" \
    --skip_video_assembly \
    data.pixel_source.downscale_when_loading="${eval_downscale_when_loading}" &
pid0=$!

CUDA_VISIBLE_DEVICES=${group1_gpu} python tools/eval.py \
    --resume_from "${group1_ckpt}" \
    --view_parallel \
    --num_view_groups 2 \
    --view_group_id 1 \
    --view_ids "${eval_group1_views}" \
    --gpu_id "${group1_gpu}" \
    --skip_video_assembly \
    data.pixel_source.downscale_when_loading="${eval_downscale_when_loading}" &
pid1=$!

wait "${pid0}"
wait "${pid1}"

echo "Merging render outputs into ${run_dir}/videos"
CUDA_VISIBLE_DEVICES=${group0_gpu} python tools/merge_view_parallel_render.py \
    --group0_dir "${run_dir}/group0/videos_eval" \
    --group1_dir "${run_dir}/group1/videos_eval" \
    --output_dir "${run_dir}/videos" \
    --view_order "${view_order}" \
    --dataset qcraft \
    --fps "${fps}" \
    --strict

echo "Merging metrics into ${run_dir}/metrics_eval"
python tools/merge_view_parallel_metrics.py \
    --group0_metrics_dir "${run_dir}/group0/metrics_eval" \
    --group1_metrics_dir "${run_dir}/group1/metrics_eval" \
    --output_dir "${run_dir}/metrics_eval" \
    --group0_views "${eval_group0_views}" \
    --group1_views "${eval_group1_views}"

if [ "${cleanup_group_eval}" = "1" ]; then
    rm -rf "${run_dir}/group0/videos_eval" "${run_dir}/group1/videos_eval"
fi

echo "Done."
echo "Group 0 checkpoint: ${group0_ckpt}"
echo "Group 1 checkpoint: ${group1_ckpt}"
echo "Unified videos: ${run_dir}/videos"
echo "Unified metrics: ${run_dir}/metrics_eval"
