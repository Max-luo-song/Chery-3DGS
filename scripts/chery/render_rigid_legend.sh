# 参数设置
################################################################################
cuda_device_id=0

clip_name="clip_1746752396800"
run_name="20250911_lidar+cam0+2downsample"
################################################################################

project_name="chery_${clip_name}"
ckpt_path="output/$project_name/$run_name/checkpoint_final.pth"
echo "Using checkpoint: $ckpt_path"

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=$cuda_device_id python tools/render_rigid_legend.py \
    --resume_from $ckpt_path \