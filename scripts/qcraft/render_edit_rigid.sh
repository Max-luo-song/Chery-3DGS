# 参数设置
################################################################################
ckpt_path="output/qcraft_20251025_163358_QCOYSD504206_1595_1610/20251125_lidar+cam1/checkpoint_final.pth"
edit_config_path="edit_config.yaml"
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/utils.sh

gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
  echo "no gpu found"
  exit 1
fi

echo "Using checkpoint: $ckpt_path"

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=${gpu} python sim_render/cam/render_edit_rigid.py \
    --resume_from $ckpt_path \
    --edit_config $edit_config_path \
    
