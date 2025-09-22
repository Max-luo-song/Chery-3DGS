# 参数设置
################################################################################
cuda_device_id=5

clip_name="clip_1746752396800"
run_name="20250915_mclidar+cam0123456+depth_loss"

traj_types=(
    # left_shift_1m
    # left_shift_3m
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
    change_lane_2m
    # change_lane_3.5m
)

cam_ids="0,1,2,3,4,5,6"

render_rgb=True
render_depth=True
generate_lidar_pc=True
################################################################################

export PYTHONPATH=$(pwd)

project_name="chery_${clip_name}"

ckpt_path="output/$project_name/$run_name/checkpoint_final.pth"
echo "Using checkpoint: $ckpt_path"

traj_types_str="$(IFS=" "; echo "${traj_types[*]}")"

CUDA_VISIBLE_DEVICES=$cuda_device_id python tools/render_novel_trajectory.py \
    --resume_from $ckpt_path \
    --traj_types $traj_types_str \
    --cam_ids $cam_ids \
    --render_depth $render_rgb \
    --render_depth $render_depth \
    --generate_lidar_pc $generate_lidar_pc
