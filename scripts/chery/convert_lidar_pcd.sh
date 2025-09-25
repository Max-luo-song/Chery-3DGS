# 参数设置
################################################################################
pcd_path="output/chery_clip_1746752396800/20250915_mclidar+cam0123456+depth_loss/novel_traj/change_lane_2m_step30000/lidar_point_clouds/frame_000190.pcd"
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/utils.sh

gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
  echo "no gpu found"
  exit 1
fi

CUDA_VISIBLE_DEVICES=${gpu} python chery_tools/convert_lidar_pcd.py \
    --pcd_path $pcd_path