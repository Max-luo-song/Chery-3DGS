# 参数设置
################################################################################
gpu=-1

ckpt_path="output/chery_clip_1746752396800/20251007_mclidar+cam0_1_2_3_4_5_6+depth_loss/checkpoint_final.pth"

traj_types=(
    # original_traj
    # left_shift_1m
    # left_shift_3m
    # left_shift_5m
    # right_shift_1m
    # right_shift_3m
    # right_shift_5m
    # change_lane_1m
    change_lane_2m
    # change_lane_3.5m
)

cam_ids=(0 1 2 3 4 5 6 7 8 9 10)
downscales=(2 2 2 2 2 2 2 2 2 2 2)
# downscales=(1 1 1 1 1 1 1 1 1 1 1)

fps=10

render_rgb=true
render_depth=false
save_images=true
generate_lidar_pc=false
################################################################################

# Pick an avaliable gpu
source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
        exit 1
    fi
fi
echo "Using GPU: ${gpu}"


echo "Using checkpoint: $ckpt_path"

bool_args=""
if [ "$render_rgb" = true ]; then
    bool_args="$bool_args --render_rgb"
fi
if [ "$render_depth" = true ]; then
    bool_args="$bool_args --render_depth"
fi
if [ "$save_images" = true ]; then
    bool_args="$bool_args --save_images"
fi
if [ "$generate_lidar_pc" = true ]; then
    bool_args="$bool_args --generate_lidar_pc"
fi

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=${gpu} python tools/render_novel_trajectory.py \
    --resume_from $ckpt_path \
    --traj_types "${traj_types[@]}" \
    --cam_ids "${cam_ids[@]}" \
    --downscales "${downscales[@]}" \
    --fps $fps \
    $bool_args