import numpy as np

from datasets.utils.box_utils import bbox_to_corner3d, get_bound_2d_mask
from datasets.utils.base_utils import project_numpy

OPENCV2DATASET = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

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
def euler_to_transform_matrix(x, y, z, yaw, pitch, roll):
    R = euler_to_rotation_matrix(yaw, pitch, roll)
    T = np.array([x, y, z])
    T_matrix = np.eye(4)
    T_matrix[:3, :3] = R
    T_matrix[:3, 3] = T
    return T_matrix


def project_points_to_image(points3d, intrinsic) -> np.ndarray:
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
    return points2d

def project_label_to_mask(dim, obj_pose, calibration_dict):
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T  # 3D bounding box in vehicle frame

    extrinsic = calibration_dict["extrinsic"]
    intrinsic = calibration_dict["intrinsic"]
    width = calibration_dict["width"]
    height = calibration_dict["height"]

    mask = get_bound_2d_mask(
        corners_3d=points_vehicle[..., :3],
        K=intrinsic,
        pose=np.linalg.inv(extrinsic),
        H=height,
        W=width,
    )
    return mask

def project_label_to_image(dim, obj2ego, cam2ego, intrinsic, img_shape):
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj2ego.T  # 3D bounding box in vehicle frame
    width, height = img_shape[1], img_shape[0]
    points_uv, valid = project_numpy(
        xyz=points_vehicle[..., :3],
        K=intrinsic,
        RT=np.linalg.inv(cam2ego),
        H=height,
        W=width,
    )
    return points_uv, valid

def convert_raw_object_type_to_class_name(obj_type) -> str:
    if obj_type == "Pedestrian":
        return "Pedestrian"

    if obj_type in ["Motorcycle", "Bicycle"]:
        return "Cyclist"

    return "Vehicle"

    
