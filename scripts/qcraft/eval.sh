# 参数设置
################################################################################
cuda_device_id=5

clip_name="20250702_133223_Q2517"
run_name="20250929_lidar+cam0_1_2_3_4_5_6_8_9_10_12"
################################################################################

project_name="qcraft_${clip_name}"
ckpt_path="output/$project_name/$run_name/checkpoint_final.pth"
echo "Using checkpoint: $ckpt_path"

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=$cuda_device_id python tools/eval.py \
    --resume_from $ckpt_path \
    # --enable_viewer