import argparse
import os
import numpy as np
import cv2
import open3d as o3d
import sys 
sys.path.append("/nas_thoru/users/guoluosong/scene_reconstruction")
from datasets.qcraft.qcraft_helpers import load_calibration, load_available_camera_ids


def load_ply_points(ply_path):
    """加载 .ply 文件并返回点云坐标和颜色"""
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found at {ply_path}")
        return None, None
    
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        print(f"Error: PLY file is empty or has no points.")
        return None, None
        
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    return points, colors


def visualize_ply_on_image(image, points, colors, intrinsics, world_to_cam):
    """
    将世界坐标系下的点云投影到图像上并可视化。
    Args:
        image (np.ndarray): 背景图像 (H, W, 3) in BGR format.
        points (np.ndarray): 点云坐标 in world frame.
        colors (np.ndarray): 点云颜色 (N, 3) in [0, 1] range.
        intrinsics (np.ndarray): 相机内参矩阵 (3, 3).
        world_to_cam (np.ndarray): 世界到相机的变换矩阵 (4, 4).
    """
    if points is None or len(points) == 0:
        return image

    # 1. 将点云转换到相机坐标系
    points_homo = np.hstack((points, np.ones((len(points), 1))))
    points_camera = np.dot(world_to_cam, points_homo.T).T[:, :3]

    # 2. 筛选在相机前方的点（z > 0）
    valid_mask = points_camera[:, 2] > 0
    if not np.any(valid_mask):
        return image # 如果没有点在相机前方，直接返回原图

    points_camera = points_camera[valid_mask]
    if colors is not None:
        valid_colors = colors[valid_mask]
    else:
        valid_colors = np.ones((len(points_camera), 3)) * [0, 0, 1] # 红色

    # 3. 投影到图像平面
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    u = (fx * points_camera[:, 0] / points_camera[:, 2] + cx).astype(np.int32)
    v = (fy * points_camera[:, 1] / points_camera[:, 2] + cy).astype(np.int32)

    # 4. 筛选在图像范围内的点
    h, w = image.shape[:2]
    valid_image_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    
    final_u = u[valid_image_mask]
    final_v = v[valid_image_mask]
    final_colors = valid_colors[valid_image_mask]

    # 5. 在图像上绘制点
    vis_image = image.copy()
    for (u, v), color in zip(zip(final_u, final_v), final_colors):
        bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        cv2.circle(vis_image, (u, v), radius=2, color=bgr_color, thickness=-1)

    return vis_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 .ply 文件中的点云投影到所有帧的图像上进行可视化。")
    parser.add_argument("--data_dir", type=str, default='data/qcraft/processed/training/20251025_163358_QCOYSD504206_1595_1610', help="Path to the input data directory.")
    parser.add_argument("--ply_path", type=str, default="debug/road_seed_pts.ply", help="Path to the .ply file to visualize.")
    parser.add_argument("--output_dir", type=str, default="debug/ply_vis", help="Path to the output directory for visualized images.")
    # --- 移除了 --frame_id 参数 ---
    parser.add_argument("--cam_id", type=int, default=0, help="Camera ID to project onto.")
    
    args = parser.parse_args()

    # --- 1. 加载所有必要的数据 ---
    ply_points, ply_colors = load_ply_points(args.ply_path)
    if ply_points is None:
        exit()

    cam_ids = load_available_camera_ids(args.data_dir)
    if args.cam_id not in cam_ids:
        print(f"Error: Camera ID {args.cam_id} not found in the dataset.")
        exit()

    extrinsics, intrinsics = load_calibration(args.data_dir, cam_ids) # cam2lidar==extrinsics
    intrinsic = intrinsics[args.cam_id]
    extrinsic = extrinsics[args.cam_id] # cam_to_ego

    # --- 2. 确定总帧数 ---
    # 通过统计 lidar_pose 文件的数量来确定总帧数
    pose_dir = os.path.join(args.data_dir, "lidar_pose")
    if not os.path.exists(pose_dir):
        print(f"Error: Lidar pose directory not found at {pose_dir}")
        exit()
    total_frames = len(os.listdir(pose_dir))
    print(f"Found {total_frames} frames to process.")

    # --- 3. 循环处理每一帧 ---
    lidar_to_world_start = np.loadtxt(
        os.path.join(args.data_dir, "lidar_pose", f"{0:06d}.txt")
    )
    cam_to_main_lidar = np.loadtxt(
        os.path.join(args.data_dir, "extrinsics", f"{args.cam_id}.txt")
    )
    for frame_id in range(total_frames):
        print(f"Processing frame {frame_id}...")
        # 加载当前帧的图像
        rgb_image_path = os.path.join(args.data_dir, "images", f"{frame_id:06d}_{args.cam_id}.png")
        if not os.path.exists(rgb_image_path):
            print(f"Warning: RGB image not found for frame {frame_id} at {rgb_image_path}. Skipping.")
            continue
        image = cv2.imread(rgb_image_path)

        lidar_to_world_current = np.loadtxt(
            os.path.join(args.data_dir, "lidar_pose", f"{frame_id:06d}.txt")
        )
        # compute lidar_to_world transformation
        lidar_to_world = (
            np.linalg.inv(lidar_to_world_start) @ lidar_to_world_current
        )
        # transformation:
        #   (opencv_cam -> qcraft_cam -> qcraft_lidar) -> current_world
        cam2world = lidar_to_world @ cam_to_main_lidar # world是第一个雷达的世界坐标系
        world2cam = np.linalg.inv(cam2world) 

        # 执行可视化
        vis_image = visualize_ply_on_image(image, ply_points, ply_colors, intrinsic, world2cam)

        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"vis_frame{frame_id:06d}_cam{args.cam_id}.jpg")
        cv2.imwrite(output_path, vis_image)

    print(f"\nAll visualizations saved to {args.output_dir}")

