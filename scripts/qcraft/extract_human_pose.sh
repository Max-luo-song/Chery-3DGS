# 参数设置
################################################################################
scene_id="20251103_134932_QLC0N1000623"
################################################################################

export PYTHONPATH=$(pwd)

export PYOPENGL_PLATFORM=osmesa
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
# export PYOPENGL_NO_FALLBACK=1

CUDA_VISIBLE_DEVICES=0 python datasets/tools/humanpose_process.py \
    --dataset qcraft \
    --data_root data/qcraft/processed/training \
    --scene_ids $scene_id