# 参数设置
################################################################################
# ckpt_path="output/chery_clip_1746752396800/20251119_mclidar+cam0+depth_loss/checkpoint_final.pth"
# ckpt_path="output/qcraft_20250819_015923_Q3707_100_130_part01/202602260852_lidar+cam0_1_2_3_5_6_7_9_10_11_12roadv1/checkpoint_final.pth"
# ckpt_path="output/qcraft_20250819_004633_Q3707_990_1010_part01/202602261156_lidar+cam0_1_2_3_5_6_7_9_10_11_12roadv1/checkpoint_final.pth"
ckpt_path="output/qcraft_20251219_100719_QCOYSD968168_3225_3240/202601210237_lidar+cam0_1_2_3_5_6_7_9_10_11_12/checkpoint_final.pth"
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
CUDA_VISIBLE_DEVICES=${gpu} python tools/eval.py \
    --resume_from $ckpt_path \
    # --enable_viewer