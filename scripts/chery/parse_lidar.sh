# 参数设置
################################################################################
path="output/chery_clip_1746752396800/20250915_mclidar+cam0123456+depth_loss/novel_traj/change_lane_2m_step30000/lidar_point_clouds/frame_000190.pcd"
################################################################################

CUDA_VISIBLE_DEVICES=$cuda_device_id python chery_tools/parse_lidar.py --path $path