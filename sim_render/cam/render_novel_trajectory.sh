#!/bin/bash

# 参数设置
################################################################################
ckpt_path="/inspire/hdd/global_user/guoluosong-253108120129/chery/new_traj/scene_reconstruction-main/output/qcraft_20250818_152739_Q3720_100_130_part01/20260506_lidar+cam0_1_2_3_5_6_7_9_10_11_12baseline/checkpoint_final.pth"

traj_types=(
    # original_traj
    # left_shift_0.2m
    # left_shift_0.4m
    # left_shift_0.6m
    # left_shift_0.8m
    left_shift_0.5m
    left_shift_1m
    left_shift_1.5m
    left_shift_2m
    left_shift_2.5m
    left_shift_3m
    # left_shift_5m
    # right_shift_1m
    # right_shift_3m
    # right_shift_5m
    # front_shift_1m
    # front_shift_3m
    # front_shift_5m
    # back_shift_1m
    # back_shift_3m
    # back_shift_5m
    # change_lane_1m
    # change_lane_2m
    # change_lane_3.5m
)

cam_ids=(0 1 2 3 5 6 7 9 10 11 12)
downscales=(1 1 1 1 1 1 1 1 1 1 1)

fps=10
render_rgb=true
render_depth=false
save_images=true
generate_lidar_pc=false

# --- DiFix3D 蒸馏修复核心配置 ---
enable_difix_distill=true
distill_ref_cam_id=0
distill_cam_ids=(1)             # 指定需要优化的相机 ID
distill_use_all_frames=true     # 开启全量帧优化
distill_max_frames=-1           # 不限制帧数（处理全部 148 帧）

# 计算总步数：148 帧 * 100 步 = 14800
distill_steps=14800
distill_lr=1e-4
distill_lr_scale=1.0

# DiFix3D 路径与设备配置
difix_src_dir="/inspire/hdd/global_user/guoluosong-253108120129/chery/new_traj/Difix3D/src"
difix_pretrained_dir="/inspire/hdd/global_user/guoluosong-253108120129/chery/new_traj/Difix3D/difix_ref"
difix_device="cuda"
difix_prompt="remove_degradation"
difix_num_inference_steps=1     # 快速修复
difix_timesteps=(199)
difix_guidance_scale=0.0
difix_use_original_traj_ref=true
difix_ref_traj_type=original_traj
# 是否使用原轨迹GT监督
difix_ref_video_path="/inspire/hdd/global_user/guoluosong-253108120129/chery/new_traj/scene_reconstruction-ga/data/refer_gt_video/scene_2_output_front_wide.mp4"
difix_trust_remote_code=false
difix_output_suffix="difix"

final_fix_ckpt="final-fix-distill.ckpt"
skip_final_fix_ckpt=true
################################################################################

source scripts/utils.sh
export PYTHONPATH=$(pwd)

# 自动选择 GPU
gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
    echo "no gpu found"
    exit 1
fi

if [ "$enable_difix_distill" = true ]; then
    if [ "$difix_use_original_traj_ref" = true ]; then
        if [ -z "$difix_ref_video_path" ]; then
            echo "ERROR: difix_use_original_traj_ref=true, but difix_ref_video_path is empty."
            exit 1
        fi
        if [ ! -f "$difix_ref_video_path" ]; then
            echo "ERROR: GT ref video not found: $difix_ref_video_path"
            exit 1
        fi
    fi
fi

echo "Using checkpoint: $ckpt_path"

cmd=(
    python sim_render/cam/render_novel_trajectory.py
    --resume_from "$ckpt_path"
    --traj_types "${traj_types[@]}"
    --cam_ids "${cam_ids[@]}"
    --downscales "${downscales[@]}"
    --fps "$fps"
)

# 添加渲染与保存标志
[ "$render_rgb" = true ] && cmd+=(--render_rgb)
[ "$render_depth" = true ] && cmd+=(--render_depth)
[ "$save_images" = true ] && cmd+=(--save_images)
[ "$generate_lidar_pc" = true ] && cmd+=(--generate_lidar_pc)

# 添加蒸馏参数
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
        --difix_ref_video_path "$difix_ref_video_path"
    )
    [ "$difix_use_original_traj_ref" = true ] && cmd+=(--difix_use_original_traj_ref)
    [ "$difix_trust_remote_code" = true ] && cmd+=(--difix_trust_remote_code)
    [ "$skip_final_fix_ckpt" = true ] && cmd+=(--skip_final_fix_ckpt)
fi

CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
