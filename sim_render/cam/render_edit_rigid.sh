# 参数设置
################################################################################
ckpt_path="/nas_thoru/users/liuzy/data/scene_reconstruction_data/10clips/output/qcraft_20250818_140332_Q3707_1250_1280_part01/20260324_lidar+cam0_1_2_3_5_6_9_10_12/checkpoint_final.pth"
edit_config_path="edit_config.yaml"
output_path_postfix="_car"
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
CUDA_VISIBLE_DEVICES=${gpu} 
python sim_render/cam/render_edit_rigid.py \
    --resume_from "${ckpt_path}" \
    --edit_config "${edit_config_path}" \
    --post_fix "${output_path_postfix}"
