import os
import matplotlib.cm as cm
import imageio
import numpy as np
import cv2


def load_lidar(data_path, lidar_type="lidar"):
    if os.path.exists(os.path.join(data_path, "lidar_pose")):
        total_frames = len(os.listdir(os.path.join(data_path, "lidar_pose")))

    lidar_points = []
    for t in range(total_frames):
        lidar_path = os.path.join(data_path, lidar_type, f"{t:03d}.bin")
        lidar_info = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 7)
        lidar_info = lidar_info[lidar_info[:, -1] == 0]
        lidar_info = lidar_info[:, :4]  # x, y, z, intensity
        lidar_points.append(lidar_info)

    return lidar_points


def read_intrinsics(data_path, num_cams=13, downscale=1):
    intrinsics_matrix = []

    for cam_id in range(num_cams):
        intrinsic = np.loadtxt(os.path.join(data_path, "intrinsics", f"{cam_id}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]

        # scale intrinsics w.r.t. load size
        fx, fy = fx / downscale, fy / downscale
        cx, cy = cx / downscale, cy / downscale

        intrinsics_matrix.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))

    return intrinsics_matrix


def read_lidar2cam_list(data_path, num_cams=13):
    lidar2cam_list = []
    for cam_id in range(num_cams):
        # load camera extrinsics
        cam_to_main_lidar = np.loadtxt(
            os.path.join(data_path, "extrinsics", f"{cam_id}.txt")
        )
        lidar2cam = np.linalg.inv(cam_to_main_lidar)
        lidar2cam_list.append(lidar2cam)

    return lidar2cam_list


def visualize_intensity(image, lidar_points, intrinsics, lidar2cam):
    points_xyz = lidar_points[:, :3]
    intensities = lidar_points[:, 3]

    # 将点云转换到相机坐标系
    points_homo = np.hstack((points_xyz, np.ones((len(points_xyz), 1))))
    points_camera = np.dot(lidar2cam, points_homo.T).T[:, :3]

    # 筛选在相机前方的点（z > 0）
    valid_mask = points_camera[:, 2] > 0
    points_camera = points_camera[valid_mask]
    valid_intensities = intensities[valid_mask]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = (fx * points_camera[:, 0] / points_camera[:, 2] + cx).astype(np.int32)
    v = (fy * points_camera[:, 1] / points_camera[:, 2] + cy).astype(np.int32)

    image_points = np.stack((u, v), axis=1)

    h, w = image.shape[:2]
    valid_image_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    image_points = image_points[valid_image_mask]
    valid_intensities = valid_intensities[valid_image_mask]

    intensities = np.clip(valid_intensities / 255, 0, 1)
    colormap = cm.get_cmap("jet")  # 获取 Jet 调色板
    colors = colormap(intensities)[:, :3]  # 获取 RGB 颜色（忽略 alpha 通道）
    colors = (colors * 255).astype(np.uint8)  # 转换为 0-255 的整数值

    for (u, v), color in zip(image_points, colors):
        bgr_color = (int(color[2]), int(color[1]), int(color[0]))
        cv2.circle(image, (u, v), 3, bgr_color, -1)

    return image


def visualize_depth_map(image, lidar_points, intrinsics, lidar2cam):
    """
    将 LiDAR 点云投影到图像上，生成深度图可视化（伪彩色）

    Args:
        image: 输入图像 (H, W, 3) BGR格式
        lidar_points: LiDAR点云 (N, 4) [x, y, z, intensity]
        intrinsics: 相机内参矩阵 (3, 3)
        lidar2cam: LiDAR到相机的外参变换矩阵 (4, 4)

    Returns:
        image_with_depth: 绘制了深度伪彩色的图像
    """
    points_xyz = lidar_points[:, :3]  # (N, 3)
    depths = points_xyz[:, 2].copy()  # 深度即为z坐标（LiDAR坐标系下）

    # 将点云转换到相机坐标系
    points_homo = np.hstack((points_xyz, np.ones((len(points_xyz), 1))))  # (N, 4)
    points_camera = (lidar2cam @ points_homo.T).T[:, :3]  # (N, 3)

    # 筛选在相机前方的点（z > 0）
    valid_mask = points_camera[:, 2] > 1e-3  # 避免除以0
    points_camera = points_camera[valid_mask]
    depths = depths[valid_mask]

    # 投影到图像平面
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = (fx * points_camera[:, 0] / points_camera[:, 2] + cx).astype(np.int32)
    v = (fy * points_camera[:, 1] / points_camera[:, 2] + cy).astype(np.int32)

    h, w = image.shape[:2]
    valid_image_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[valid_image_mask]
    v = v[valid_image_mask]
    depths = depths[valid_image_mask]

    # 创建深度图画布（初始化为背景）
    depth_vis = image.copy()

    # 归一化深度用于颜色映射（可根据场景调整范围）
    depth_min, depth_max = depths.min(), depths.max()
    if depth_max > depth_min:
        normalized_depth = (depths - depth_min) / (depth_max - depth_min)
    else:
        normalized_depth = np.zeros_like(depths)

    # 使用 jet 色图
    colormap = cm.get_cmap("jet")
    colors = colormap(normalized_depth)[:, :3]  # (N, 3) RGB in [0,1]
    colors = (colors * 255).astype(np.uint8)  # 转为 0-255

    # 绘制每个点（可改为更大半径或填充）
    for (ui, vi), color in zip(zip(u, v), colors):
        bgr_color = (int(color[2]), int(color[1]), int(color[0]))  # RGB -> BGR
        cv2.circle(depth_vis, (ui, vi), radius=2, color=bgr_color, thickness=-1)

    return depth_vis


if __name__ == "__main__":
    data_path = "data/chery/processed/training/clip_1746752396800"
    output_dir = "output/chery_clip_1746752396800"
    num_cams = 7

    depth_save_dir = os.path.join(output_dir, "depths_vis")
    lidar_save_dir = os.path.join(output_dir, "intensity_vis")
    os.makedirs(depth_save_dir, exist_ok=True)
    os.makedirs(lidar_save_dir, exist_ok=True)

    lidar_points_list = load_lidar(data_path=data_path, lidar_type="mclidar")
    intrinsics_list = read_intrinsics(data_path=data_path, num_cams=num_cams)
    lidar2cam_list = read_lidar2cam_list(data_path=data_path, num_cams=num_cams)

    for cam_id in range(num_cams):
        print(f"Visualizing lidar for cam {cam_id}...")

        video_path = os.path.join(depth_save_dir, f"cam_{cam_id}.mp4")
        lidar_writer = imageio.get_writer(video_path, mode="I", fps=10)
        depth_writer = imageio.get_writer(video_path, mode="I", fps=10)

        intrinsics = intrinsics_list[cam_id]
        lidar2cam = lidar2cam_list[cam_id]

        for frame_id in range(len(lidar_points_list)):
            lidar_points = lidar_points_list[frame_id]
            rgb_image_path = os.path.join(
                data_path, "images", f"{frame_id:03d}_{cam_id}.jpg"
            )
            image = cv2.imread(rgb_image_path)

            # Visualize depth map >>>
            vis_depth_image = visualize_depth_map(
                image.copy(), lidar_points, intrinsics, lidar2cam
            )

            depth_save_path = os.path.join(
                depth_save_dir, f"{frame_id:03d}_{cam_id}.jpg"
            )
            cv2.imwrite(depth_save_path, vis_depth_image)

            depth_writer.append_data(cv2.cvtColor(vis_depth_image, cv2.COLOR_BGR2RGB))
            # <<<

            # Visualize LiDAR intensity image >>>
            vis_lidar_image = visualize_intensity(
                image.copy(), lidar_points, intrinsics, lidar2cam
            )

            image_save_path = os.path.join(
                lidar_save_dir, f"{frame_id:03d}_{cam_id}.jpg"
            )
            cv2.imwrite(image_save_path, vis_lidar_image)

            lidar_writer.append_data(cv2.cvtColor(vis_lidar_image, cv2.COLOR_BGR2RGB))
            # <<<

        depth_writer.close()
        lidar_writer.close()