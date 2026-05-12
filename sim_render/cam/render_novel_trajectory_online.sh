#!/bin/bash

# DiFix3D online repair + progressive mixed distillation.
# Default curriculum (11 cams): left 1.5m -> original -> left 3m -> original.

################################################################################
ckpt_path=${ckpt_path:-"/inspire/hdd/global_user/guoluosong-253108120129/chery/new_traj/scene_reconstruction-main/output/qcraft_20250818_152739_Q3720_100_130_part01/20260506_lidar+cam0_1_2_3_5_6_7_9_10_11_12baseline/checkpoint_final.pth"}
final_fix_ckpt_suffix=${final_fix_ckpt_suffix:-"difix_prog_left1p5_ori_left3_ori"}
difix_output_suffix=${difix_output_suffix:-"difix_left1p5_ori_left3_ori"}

traj_types=(
    left_shift_1.5m
    original_traj
    left_shift_3m
    original_traj
)
distill_stage_repeats=(30 10 30 10)  # 可调超参数：每个阶段“每视角重复次数”
distill_ref_traj_as_original_stage=true

cam_ids=(0 1 2 3 5 6 7 9 10 11 12)
downscales=(1 1 1 1 1 1 1 1 1 1 1)

fps=10
render_rgb=true
render_depth=false
save_images=true
generate_lidar_pc=false

enable_difix_distill=true
distill_ref_cam_id=0
distill_cam_ids=(0 1 2 3 5 6 7 9 10 11 12)
distill_use_all_frames=true
distill_max_frames=-1

# Keep the existing distillation setting unchanged.
distill_steps=14800
distill_lr=1e-4
distill_lr_scale=1.0

# Ratio mode is kept for backward compatibility; progressive stage repeats are preferred.
distill_mix_original=false
distill_original_sample_ratio=0.65

difix_src_dir=${difix_src_dir:-"/inspire/hdd/global_user/guoluosong-253108120129/chery/new_traj/Difix3D/src"}
difix_pretrained_dir=${difix_pretrained_dir:-"/inspire/hdd/global_user/guoluosong-253108120129/chery/new_traj/Difix3D/difix_ref"}
difix_device="cuda"
difix_prompt="remove_degradation"
difix_num_inference_steps=1
difix_timesteps=(199)
difix_guidance_scale=0.0
difix_use_original_traj_ref=true
difix_ref_traj_type=original_traj
generate_ref_videos=true
overwrite_ref_videos=false
difix_ref_video_dir=${difix_ref_video_dir:-"/inspire/hdd/global_user/guoluosong-253108120129/chery/new_traj/scene_reconstruction-ga/data/refer_gt_video"}
difix_trust_remote_code=false

eval_original_traj_after_distill=true
eval_postfix=${eval_postfix:-"prog_mix_original_traj"}
eval_render_full=true
eval_render_test=false

final_fix_ckpt="checkpoint_final_${final_fix_ckpt_suffix}.pth"
skip_final_fix_ckpt=false
################################################################################

source scripts/utils.sh
export PYTHONPATH=$(pwd)

gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
    echo "no gpu found"
    exit 1
fi

if [ "$enable_difix_distill" = true ]; then
    if [ "$difix_use_original_traj_ref" = true ]; then
        if [ -z "$difix_ref_video_dir" ]; then
            echo "ERROR: difix_use_original_traj_ref=true, but difix_ref_video_dir is empty."
            exit 1
        fi
        if [ "$generate_ref_videos" = true ]; then
            ref_cmd=(
                python sim_render/cam/export_reference_videos.py
                --resume_from "$ckpt_path"
                --cam_ids "${cam_ids[@]}"
                --downscales "${downscales[@]}"
                --fps "$fps"
                --output_dir "$difix_ref_video_dir"
            )
            [ "$overwrite_ref_videos" = true ] && ref_cmd+=(--overwrite)
            CUDA_VISIBLE_DEVICES="${gpu}" "${ref_cmd[@]}"
        else
            for cam_id in "${distill_cam_ids[@]}"; do
                if [ ! -f "${difix_ref_video_dir}/cam${cam_id}.mp4" ]; then
                    echo "ERROR: GT ref video not found: ${difix_ref_video_dir}/cam${cam_id}.mp4"
                    exit 1
                fi
            done
        fi
    fi
fi

echo "Using checkpoint: $ckpt_path"
echo "Mixed distillation original replay ratio: ${distill_original_sample_ratio}"

cmd=(
    python sim_render/cam/render_novel_trajectory.py
    --resume_from "$ckpt_path"
    --traj_types "${traj_types[@]}"
    --cam_ids "${cam_ids[@]}"
    --downscales "${downscales[@]}"
    --fps "$fps"
)

[ "$render_rgb" = true ] && cmd+=(--render_rgb)
[ "$render_depth" = true ] && cmd+=(--render_depth)
[ "$save_images" = true ] && cmd+=(--save_images)
[ "$generate_lidar_pc" = true ] && cmd+=(--generate_lidar_pc)

if [ "$enable_difix_distill" = true ]; then
    cmd+=(
        --enable_difix_distill
        --distill_ref_cam_id "$distill_ref_cam_id"
        --distill_cam_ids "${distill_cam_ids[@]}"
        --distill_use_all_frames
        --distill_max_frames "$distill_max_frames"
        --distill_steps "$distill_steps"
        --distill_lr "$distill_lr"
        --distill_lr_scale "$distill_lr_scale"
        --final_fix_ckpt "$final_fix_ckpt"
        --difix_src_dir "$difix_src_dir"
        --difix_pretrained_dir "$difix_pretrained_dir"
        --difix_device "$difix_device"
        --difix_prompt "$difix_prompt"
        --difix_num_inference_steps "$difix_num_inference_steps"
        --difix_timesteps "${difix_timesteps[@]}"
        --difix_guidance_scale "$difix_guidance_scale"
        --difix_output_suffix="$difix_output_suffix"
        --difix_ref_traj_type "$difix_ref_traj_type"
        --difix_ref_video_dir "$difix_ref_video_dir"
    )
    if [ ${#distill_stage_repeats[@]} -gt 0 ]; then
        cmd+=(--distill_stage_repeats "${distill_stage_repeats[@]}")
    fi
    [ "$distill_ref_traj_as_original_stage" = true ] && cmd+=(--distill_ref_traj_as_original_stage)
    [ "$distill_mix_original" = true ] && cmd+=(--distill_mix_original --distill_original_sample_ratio "$distill_original_sample_ratio")
    [ "$difix_use_original_traj_ref" = true ] && cmd+=(--difix_use_original_traj_ref)
    [ "$difix_trust_remote_code" = true ] && cmd+=(--difix_trust_remote_code)
    [ "$skip_final_fix_ckpt" = true ] && cmd+=(--skip_final_fix_ckpt)
fi

CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"

if [ "$eval_original_traj_after_distill" = true ] && [ "$skip_final_fix_ckpt" = false ]; then
    if [[ "${final_fix_ckpt}" = /* ]]; then
        final_ckpt_path="${final_fix_ckpt}"
    else
        final_ckpt_path="$(dirname "${ckpt_path}")/${final_fix_ckpt}"
    fi
    if [ ! -f "${final_ckpt_path}" ]; then
        echo "ERROR: final checkpoint not found: ${final_ckpt_path}"
        exit 1
    fi
    gpu="${gpu}" \
    ckpt_path="${final_ckpt_path}" \
    eval_postfix="${eval_postfix}" \
    render_full="${eval_render_full}" \
    render_test="${eval_render_test}" \
    bash scripts/qcraft/cal_new_traj_pref.sh
fi
