# 参数设置
################################################################################
cuda_device_id=0

clip_name="clip_1746752396800"
run_name="20250911_lidar+cam0+2downsample"

rigid_id=1
edit_value=(3 0 0) 
################################################################################

project_name="chery_${clip_name}"
ckpt_path="output/$project_name/$run_name/checkpoint_final.pth"
echo "Using checkpoint: $ckpt_path"

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=$cuda_device_id python tools/render_edit_rigid.py \
    --resume_from $ckpt_path \
    --rigid_id $rigid_id \
    --edit_value ${edit_value[@]} \
    
