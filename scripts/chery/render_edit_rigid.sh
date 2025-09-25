# 参数设置
################################################################################
clip_name="clip_1746752396800"
run_name="20250911_lidar+cam0+2downsample"

rigid_id=1
edit_value=(3 0 0) 
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/utils.sh

gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
  echo "no gpu found"
  exit 1
fi

project_name="chery_${clip_name}"
ckpt_path="output/$project_name/$run_name/checkpoint_final.pth"
echo "Using checkpoint: $ckpt_path"

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=${gpu} python tools/render_edit_rigid.py \
    --resume_from $ckpt_path \
    --rigid_id $rigid_id \
    --edit_value ${edit_value[@]} \
    
