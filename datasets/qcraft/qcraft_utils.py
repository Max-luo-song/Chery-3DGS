import numpy as np
import cv2

from scipy.spatial.transform import Rotation


def euler_to_rotation_matrix(yaw, pitch, roll):
    # Z 轴旋转（yaw）
    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    # Y 轴旋转（pitch）
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    # X 轴旋转（roll）
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    R = R_z @ R_y @ R_x
    return R


# 将欧拉角和位置转换为4x4变换矩阵
def pose_to_transform_matrix(x, y, z, yaw, pitch, roll):
    R = euler_to_rotation_matrix(yaw, pitch, roll)
    T = np.array([x, y, z])
    T_matrix = np.eye(4)
    T_matrix[:3, :3] = R
    T_matrix[:3, 3] = T
    return T_matrix


def project_points_to_image(points3d, intrinsic, img_shape):
    # 将3D点转换为齐次坐标
    points3d_homogeneous = np.concatenate(
        [points3d, np.ones((points3d.shape[0], 1))], axis=1
    )
    # 投影到2D
    camera_intrinsic_extended = np.hstack([intrinsic, np.zeros((3, 1))])
    points2d_homogeneous = camera_intrinsic_extended @ points3d_homogeneous.T
    # 转换为非齐次坐标
    z = points2d_homogeneous[2, :]
    points2d = points2d_homogeneous[:2, :] / z
    # 添加有效性检查
    valid_mask = z > 0
    points2d = points2d[:, valid_mask].T

    # # 检查 2D 点是否在图像范围内
    # h, w = img_shape
    # valid_mask[valid_mask] &= (
    #     (points2d[:, 0] >= 0)
    #     & (points2d[:, 0] < w)
    #     & (points2d[:, 1] >= 0)
    #     & (points2d[:, 1] < h)
    # )
    # points2d = points2d[valid_mask[valid_mask]]
    return points2d


def draw_and_fill_box(img, points2d):
    # 计算最小外接矩形
    x, y, w, h = cv2.boundingRect(points2d.astype(int))
    # print("points2d:", points2d)
    # print(f"Bounding Rect: x={x}, y={y}, w={w}, h={h}")

    # 绘制并填充矩形
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return img


def filter_points_in_box(pointcloud, obj_center_pos, size):
    """
    筛选出以obj_center_pos为中心、尺寸为size的矩形区域内的点云

    参数:
        pointcloud: numpy数组，形状为(N, 3)，表示点云数据
        obj_center_pos: 列表或数组，表示矩形中心坐标[x, y, z]
        size: 列表或数组，表示矩形在x、y、z三个维度上的尺寸[l, w, h]

    返回:
        筛选后的点云数据
    """
    if pointcloud is None:
        return None

    # 计算矩形边界
    half_size = np.array(size) / 2
    min_bounds = np.array(obj_center_pos) - half_size
    max_bounds = np.array(obj_center_pos) + half_size

    # 筛选在矩形边界内的点
    mask = (
        (pointcloud[:, 0] >= min_bounds[0])
        & (pointcloud[:, 0] <= max_bounds[0])
        & (pointcloud[:, 1] >= min_bounds[1])
        & (pointcloud[:, 1] <= max_bounds[1])
        & (pointcloud[:, 2] >= min_bounds[2])
        & (pointcloud[:, 2] <= max_bounds[2])
    )

    return pointcloud[mask]
