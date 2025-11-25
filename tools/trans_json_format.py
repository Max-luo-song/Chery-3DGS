
import json
import numpy as np
import math
import os
from pathlib import Path
import time

def focal2fov(focal, pixels):
    """将焦距转换为视场角"""
    return 2 * math.atan(pixels / (2 * focal))

def load_ego_poses(pose_dir):
    """加载ego poses"""
    poses = {}
    pose_files = sorted(Path(pose_dir).glob("*.txt"))
    
    for pose_file in pose_files:
        frame_idx = int(pose_file.stem)
        pose_matrix = np.loadtxt(pose_file)
        poses[frame_idx] = pose_matrix
    
    return poses

def load_calibration(extrinsics_dir, intrinsics_dir):
    """加载相机标定参数"""
    # 加载外参
    extrinsics = {}
    cam_id_list = [0]
    for cam_id in cam_id_list:
        ext_file = Path(extrinsics_dir) / f"{cam_id}.txt"
        if ext_file.exists():
            extrinsics[cam_id] = np.loadtxt(ext_file)
    
    # 加载内参
    intrinsics = {}
    cam_id_list = [0]
    intrinsics = {}
    for cam_id in cam_id_list:
        int_file = Path(intrinsics_dir) / f"{cam_id}.txt"
        if int_file.exists():
            with open(int_file, 'r') as f:
                lines = f.readlines()
                # 读取前四行，每行一个值
                fx = float(lines[0].strip())
                fy = float(lines[1].strip())
                cx = float(lines[2].strip())
                cy = float(lines[3].strip())
                
                intrinsics[cam_id] = {
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy
                }
    
    return extrinsics, intrinsics

def get_lane_shift_direction(ego_frame_poses, frame):
    """获取车道偏移方向（简化版本）"""
    # 这里需要根据实际需求实现
    # 返回偏移方向向量
    return np.array([1.0, 0.0, 0.0])  # 默认X方向偏移

def generate_camera_json(base_dir, scene_id, cam_id, frame_idx, shift=3.0):
    """
    生成相机参数JSON
    
    Args:
        base_dir: 基础目录路径
        scene_id: 场景ID (如 "20250702_133223_Q2517")
        cam_id: 相机ID (0 或 1)
        frame_idx: 帧索引
        shift: 偏移量
    """
    
    # 构建路径
    pose_dir = os.path.join(base_dir, "data", "qcraft", "processed", "training", scene_id, "lidar_pose")
    extrinsics_dir = os.path.join(base_dir, "data", "qcraft", "processed", "training", scene_id, "extrinsics")
    intrinsics_dir = os.path.join(base_dir, "data", "qcraft", "processed", "training", scene_id, "intrinsics")
    
    # 加载数据
    ego_cam_poses = load_ego_poses(pose_dir)
    extrinsics, intrinsics = load_calibration(extrinsics_dir, intrinsics_dir)
    
    # 获取ego pose
    ego_pose = ego_cam_poses[frame_idx]
    ego_pose_shift = ego_pose.copy()
    # print("ego_pose_shift:", ego_pose_shift)
    # time.sleep(10)
    # 计算车道偏移
    lane_shift_direction = get_lane_shift_direction(ego_cam_poses, frame_idx)
    lane_shift_sign = 1  # 根据场景确定符号
    ego_pose_shift[:3, 3] += lane_shift_sign * lane_shift_direction * shift
    
    # 计算c2w和w2c
    c2w = ego_pose_shift @ extrinsics[cam_id]
    w2c = np.linalg.inv(c2w)
    
    # 提取R和T
    R = w2c[:3, :3].tolist()
    T = w2c[:3, 3].tolist()
    
    # 获取内参
    intrinsic_params = intrinsics[cam_id]
    fx = intrinsic_params['fx']
    fy = intrinsic_params['fy']
    cx = intrinsic_params['cx']
    cy = intrinsic_params['cy']
    
    # 构建内参矩阵K
    K = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]
    
    # 计算视场角
    image_width = 1024
    image_height = 512
    FoVx = focal2fov(fx, image_width)
    FoVy = focal2fov(fy, image_height)
    
    # 构建guidance_rgb_path
    guidance_rgb_path = f"{scene_id}/lidar/color_render_shift_{shift:.2f}/{frame_idx:06d}_0.png"
    
    # 构建JSON数据
    camera_data = {
        "cam_id": frame_idx,
        "R": R,
        "T": T,
        "FoVx": FoVx,
        "FoVy": FoVy,
        "K": K,
        "guidance_rgb_path": guidance_rgb_path
    }
    
    return camera_data

def batch_generate_camera_jsons(base_dir, scene_id, output_dir):
    """批量生成相机参数JSON文件"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载所有poses来确定帧范围
    pose_dir = os.path.join(base_dir, "data", "qcraft", "processed", "training", scene_id, "lidar_pose")
    ego_cam_poses = load_ego_poses(pose_dir)
    # print("ego_cam_poses.keys():", ego_cam_poses.keys())
    # time.sleep(10)
    cam_id_list = [0]
    # 为每个相机和每帧生成JSON
    for cam_id in cam_id_list:
        for frame_idx in ego_cam_poses.keys():
            camera_data = generate_camera_json(base_dir, scene_id, cam_id, frame_idx)
            print("camera_data:", camera_data)
            # 保存JSON文件
            output_filename = f"camera_{cam_id}_frame_{frame_idx:06d}.json"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w') as f:
                json.dump(camera_data, f, indent=4)
            
            print(f"Generated: {output_path}")
                

# 使用示例
if __name__ == "__main__":
    # 配置参数
    base_dir = "/data4/gls/code/chery_scene_reconstruction"
    scene_id = "20250702_133223_Q2517"
    output_dir = "./novel_caminfos"
    
    # 生成单个JSON示例
    # try:
    #     camera_data = generate_camera_json(base_dir, scene_id, cam_id=0, frame_idx=0)
    #     print(json.dumps(camera_data, indent=4))
    # except Exception as e:
    #     print(f"Error: {e}")
    
    # 批量生成所有JSON文件
    batch_generate_camera_jsons(base_dir, scene_id, output_dir)
