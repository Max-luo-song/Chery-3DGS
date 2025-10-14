# 参数设置
################################################################################
gpu=-1

ckpt_path="output/chery_clip_1746752396800/20250930_mclidar+cam0123456+depth_loss/checkpoint_final.pth"

rigid_id=1
edit_value=(3 0 0) 
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

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=${gpu} python tools/render_edit_rigid.py \
    --resume_from $ckpt_path \
    --rigid_id $rigid_id \
    --edit_value ${edit_value[@]} \
    
