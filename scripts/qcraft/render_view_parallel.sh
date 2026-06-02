#!/usr/bin/env bash
set -euo pipefail

################################################################################
scene_idx="${scene_idx:-20250818_152739_Q3720_100_130_part01}"
lidar_type="${lidar_type:-lidar}"
extra_config_info="${extra_config_info:-baseline}"

output_root="${output_root:-output}"
project_name="${project_name:-qcraft_${scene_idx}}"
date_str="20260529"
run_name="${run_name:-${date_str}_${lidar_type}+traincam0_1_5_9_12_2_3_6_10_015912_23610}"

group0_gpu="${group0_gpu:-0}"
group1_gpu="${group1_gpu:-1}"
eval_group0_views="${eval_group0_views:-0,1,5,9,7,12}"
eval_group1_views="${eval_group1_views:-2,3,6,10,11}"
eval_downscale_when_loading="${eval_downscale_when_loading:-[1]}"
view_order="${view_order:-0,1,2,3,5,6,7,9,10,11,12}"
fps="${fps:-10}"
################################################################################

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${repo_root}/.torch_extensions}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

run_dir="${output_root}/${project_name}/${run_name}"
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

echo "Run directory: ${run_dir}"
echo "Rendering group 0 on GPU ${group0_gpu}: ${eval_group0_views}"
echo "Rendering group 1 on GPU ${group1_gpu}: ${eval_group1_views}"

echo "Preparing gsplat CUDA extension cache: ${TORCH_EXTENSIONS_DIR}"
CUDA_VISIBLE_DEVICES=${group0_gpu} python -c "from gsplat.cuda._backend import _C; assert _C is not None"

CUDA_VISIBLE_DEVICES=${group0_gpu} python tools/eval.py \
    --resume_from "${group0_ckpt}" \
    --eval_output_dir "${run_dir}/group0" \
    --view_parallel \
    --num_view_groups 2 \
    --view_group_id 0 \
    --view_ids "${eval_group0_views}" \
    --gpu_id "${group0_gpu}" \
    data.pixel_source.downscale_when_loading="${eval_downscale_when_loading}" &

pid0=$!

CUDA_VISIBLE_DEVICES=${group1_gpu} python tools/eval.py \
    --resume_from "${group1_ckpt}" \
    --eval_output_dir "${run_dir}/group1" \
    --view_parallel \
    --num_view_groups 2 \
    --view_group_id 1 \
    --view_ids "${eval_group1_views}" \
    --gpu_id "${group1_gpu}" \
    data.pixel_source.downscale_when_loading="${eval_downscale_when_loading}" &

pid1=$!

wait "${pid0}"
wait "${pid1}"

echo "Merging group render outputs into ${run_dir}/videos"
CUDA_VISIBLE_DEVICES=${group0_gpu} python tools/merge_view_parallel_render.py \
    --group0_dir "${run_dir}/group0/videos_eval" \
    --group1_dir "${run_dir}/group1/videos_eval" \
    --output_dir "${run_dir}/videos" \
    --view_order "${view_order}" \
    --dataset qcraft \
    --fps "${fps}" \
    --strict

echo "Merging group metrics into ${run_dir}/metrics_eval"
python tools/merge_view_parallel_metrics.py \
    --group0_metrics_dir "${run_dir}/group0/metrics_eval" \
    --group1_metrics_dir "${run_dir}/group1/metrics_eval" \
    --output_dir "${run_dir}/metrics_eval" \
    --group0_views "${eval_group0_views}" \
    --group1_views "${eval_group1_views}"

echo "Unified render outputs saved to: ${run_dir}/videos"
echo "Unified metrics saved to: ${run_dir}/metrics_eval"
