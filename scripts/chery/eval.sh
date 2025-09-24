# 参数设置
################################################################################
cuda_device_id=0

clip_name="clip_1746752396800"
run_name="20250915_mclidar+cam0123456+depth_loss"
################################################################################

project_name="chery_${clip_name}"
ckpt_path="output/$project_name/$run_name/checkpoint_final.pth"
echo "Using checkpoint: $ckpt_path"

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=$cuda_device_id python tools/eval.py \
    --resume_from $ckpt_path \
    # --enable_viewer