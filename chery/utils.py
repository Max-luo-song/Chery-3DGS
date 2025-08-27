import open3d as o3d
import numpy as np
import struct
import yaml
import os
from PIL import Image
import cv2
import time
import json
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation

def imu2ego():
    """
    imu在后轴中心，右前上 -> 车体坐标系在后轴中心地面点，前左上
    1) 右x，前y，上z -> 前x ，左y， 上z
    2) z平移到地面，E03轮半径: 0.349m
    """
    T = np.eye(4)
    r = R.from_euler('z', -90, degrees=True)
    T[:3, :3] = r.as_matrix()
    T[2, 3] = 0.349  # E03轮半径
    return T

def load_extrinsic_yaml(extrinsics_yaml):
    with open(extrinsics_yaml, "r", encoding="utf-8") as ff:
        extrin = yaml.safe_load(ff)
    trans_xyz = [extrin['transform']['translation'][ii] 
                 for ii in ['x', 'y', 'z']]
    rot_xyzw = [extrin['transform']['rotation'][ii] 
                for ii in ['x', 'y', 'z', 'w']]
    rot_mat = R.from_quat(rot_xyzw).as_matrix()
    extrin_mat = np.eye(4, dtype=np.float64)
    extrin_mat[:3, :3] = rot_mat
    extrin_mat[:3, 3] = np.float64(trans_xyz)
    return extrin_mat

def load_lidar2camera_yaml(extrinsics_yaml):
    with open(extrinsics_yaml, "r", encoding="utf-8") as ff:
        extrin = yaml.safe_load(ff)
    extrin_mat = np.float64(extrin['transform'])
    return extrin_mat

def load_intrinsic_yaml(intrinsics_yaml):
    with open(intrinsics_yaml, "r", encoding="utf-8") as ff:
        intrin = yaml.safe_load(ff)
    distort_f = intrin['D']
    intrin_9nums = intrin['K']  # for 3x3 mat
    intrin_12nums = intrin['P']  # for 3x4 mat
    height = intrin['height']
    width = intrin['width']
    return distort_f, intrin_9nums, intrin_12nums, height, width

def load_camera_intrinsic_yaml(intrinsics_yaml):
    with open(intrinsics_yaml, "r", encoding="utf-8") as ff:
        intrin = yaml.safe_load(ff)
    distort_f = intrin['D']  # 5 params
    intrin_4nums = intrin['K']  # 4 params
    intrin_9nums = [
        intrin_4nums[0], 0, intrin_4nums[2],
        0, intrin_4nums[1], intrin_4nums[3],
        0, 0, 1
    ]
    intrin_12nums = [
        intrin_4nums[0], 0, intrin_4nums[2], 0,
        0, intrin_4nums[1], intrin_4nums[3], 0,
        0, 0, 1, 0,
    ]
    height = intrin['height']
    width = intrin['width']
    return distort_f, intrin_9nums, intrin_12nums, height, width

def read_pcd_file(filepath):
    try:
        point_cloud = o3d.io.read_point_cloud(filepath)
        return np.asarray(point_cloud.points)
    except Exception as e:
        print(f"Error reading PCD file: {e}")
        return None

def write_bin_file(bin_filepath, pointcloud):
    """将点云数据列表写入Bin文件"""
    with open(bin_filepath, 'wb') as file:
        for x, y, z in pointcloud:
            # 将x, y, z坐标转换为二进制数据并写入文件
            file.write(struct.pack('fff', x, y, z))

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
        (pointcloud[:, 0] >= min_bounds[0]) & (pointcloud[:, 0] <= max_bounds[0]) &
        (pointcloud[:, 1] >= min_bounds[1]) & (pointcloud[:, 1] <= max_bounds[1]) &
        (pointcloud[:, 2] >= min_bounds[2]) & (pointcloud[:, 2] <= max_bounds[2])
    )
    
    return pointcloud[mask]

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def yaml_to_transform_matrix(data):
    transform_data = data['transform']
    transform_matrix = np.array(transform_data)
    return transform_matrix

def read_lidar2camera(path):
    lidar2camera = []
    files_path = []
    lidar2camera_name = ['lidar2frontwide.yaml', 'lidar2frontmain.yaml', 'lidar2leftfront.yaml',
                         'lidar2leftrear.yaml', 'lidar2rightfront.yaml', 'lidar2rightrear.yaml', 'lidar2rearmain.yaml']
    for i in range(len(lidar2camera_name)):
        files_path.append(os.path.join(path, lidar2camera_name[i]))
    for file_path in files_path:
        yaml_data = read_yaml_file(file_path)
        transform_matrix = yaml_to_transform_matrix(yaml_data)
        lidar2camera.append(transform_matrix)
    return lidar2camera

def read_matrix_from_txt(filepath):
    with open(filepath, 'r') as file:
        # 读取前四行
        lines = [next(file) for _ in range(4)]
        # 去除每行的前后空白（如果有的话）
        lines = [line.strip() for line in lines]
        # 将每行转换为浮点数
        fx, fy, cx, cy = [float(line) for line in lines]
    
    # 构造3x3内参矩阵
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    # print("intrinsic_matrix:", intrinsic_matrix)
    # time.sleep(10000)
    
    return intrinsic_matrix

def read_extrinsic_matrix(file_path):
    """
    读取包含外参矩阵的txt文件并返回4x4的numpy数组
    
    参数:
        file_path: txt文件的路径
        
    返回:
        4x4的外参矩阵(numpy数组)
    """
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            # 将每行的字符串分割为浮点数列表
            row = [float(x) for x in line.strip().split()]
            matrix.append(row)
    
    # 转换为numpy数组
    extrinsic_matrix = np.array(matrix)
    
    # 验证矩阵形状是否为4x4
    if extrinsic_matrix.shape != (4, 4):
        raise ValueError(f"读取的矩阵形状为{extrinsic_matrix.shape}，不是4x4矩阵")
    
    return extrinsic_matrix

def read_camera_intrinsic(path):
    camera_intrinsic = []
    files_path = []
    intrinsic_name = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt']
    for i in range(len(intrinsic_name)):
        files_path.append(os.path.join(path, intrinsic_name[i]))
    for file_path in files_path:
        matrix = read_matrix_from_txt(file_path)
        camera_intrinsic.append(matrix)
    return camera_intrinsic

def read_camera_extrinsics(path):
    camera_extrinsics = []
    files_path = []
    extrinsics_name = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt']
    for i in range(len(extrinsics_name)):
        files_path.append(os.path.join(path, extrinsics_name[i]))
    for file_path in files_path:
        matrix = read_extrinsic_matrix(file_path)
        camera_extrinsics.append(matrix)
    return camera_extrinsics

def project_to_image(points, lidar2camera, camera_intrinsics):
    """
    将点云中的点通过lidar2camera矩阵和相机内参转换到图像坐标系中。
    """
    # 将点云转换为齐次坐标
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    # 点云到相机坐标系的转换
    points_camera = lidar2camera @ points_homogeneous.T
    # 相机坐标系到图像坐标系的转换
    points_image = camera_intrinsics @ points_camera[:3, :]
    # 将图像坐标系的z坐标归一化
    points_image = points_image / points_image[2, :]
    return points_image[:2, :].T

def generate_dynamic_mask(points2d, image_size, image):
    """
    根据图像坐标系中的点生成dynamic_mask标签。
    """
    image_with_box = draw_and_fill_box(image, points2d)
    
    return image_with_box

# def generate_dynamic_mask(points2d, image_size, image):
#     """
#     根据图像坐标系中的点生成dynamic_mask标签。
#     """
#     # 如果没有提供图像，初始化为黑色图像
#     if image is None:
#         image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
#     # 过滤有效点
#     valid_points = points2d[
#         (points2d[:, 0] >= 0) & (points2d[:, 0] < image_size[1]) & 
#         (points2d[:, 1] >= 0) & (points2d[:, 1] < image_size[0])
#     ]
    
#     # 只有当有有效点时才绘制框
#     if len(valid_points) > 0:
#         image_with_box = draw_and_fill_box(image.copy(), valid_points)
#     else:
#         image_with_box = image.copy()
    
#     return image_with_box

def get_image_size(image_path):
    """
    读取指定路径的jpg图片，返回图片尺寸。
    
    :param image_path: 图片的文件路径
    :return: 图片尺寸，例如(1080, 1920)
    """
    with Image.open(image_path) as img:
        image_size = img.size
        adjusted_size = (image_size[1], image_size[0])
        return adjusted_size
    
def project_points_to_image(points3d, camera_intrinsic):
    # 将3D点转换为齐次坐标
    points3d_homogeneous = np.concatenate([points3d, np.ones((points3d.shape[0], 1))], axis=1)
    # 投影到2D
    camera_intrinsic_extended = np.hstack([camera_intrinsic, np.zeros((3, 1))])
    points2d_homogeneous = camera_intrinsic_extended @ points3d_homogeneous.T
    # 转换为非齐次坐标
    points2d = points2d_homogeneous[:2, :] / points2d_homogeneous[2, :]
     # 添加有效性检查
    valid_mask = points2d_homogeneous[2, :] > 0  # 只保留z>0的点
    points2d = points2d[:, valid_mask]   
    return points2d.T

def draw_and_fill_box(image, points2d):
    # 计算最小外接矩形
    x, y, w, h = cv2.boundingRect(points2d.astype(int))
    # 绘制并填充矩形
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return image

def find_track_id_frame(track_id, result):
    frame_list = []
    for frame, track_ids in result.items():
        # print("track_id:", track_id)
        # print("track_ids:", track_ids)
        # print("frame:", frame)
        # print(f"track_id: {track_id}, type: {type(track_id)}")
        if int(track_id) in track_ids:
            frame_list.append(int(frame))
        # time.sleep(10000)
    return frame_list

def find_track_id_obj2world(track_id, data, frame_list, source_dir, clip_name, ex_dir_extrinsics):
    
    obj2world_list = []
    lidar2camera = read_lidar2camera(os.path.join(source_dir, clip_name, "extrinsics", "lidar2camera"))[0]
    print("ex_dir_extrinsics:", ex_dir_extrinsics)
    cam2world = read_camera_extrinsics(ex_dir_extrinsics)[0]
    ### 每一个物体在每一个时间戳之内的obj2world
    for frame_id in range(len(frame_list)):
        frame_data = data['frames'][frame_id]
        object_detection_anns_info = frame_data.get('annotated_info', {}).get(
            '3d_city_object_detection_annotated_info', {}
        ).get('annotated_info', {}).get('3d_object_detection_info', {}).get(
            '3d_object_detection_anns_info', [])   
        ### frame_data中还是有很多track
        for i in range(len(object_detection_anns_info)):
            if object_detection_anns_info[i]['track_id'] == int(track_id):
                each_object = object_detection_anns_info[i]  
                #### each_object是指定的track_id
                l, w, h = each_object['size']  # ann_box is one of dynamic objs
                box2lidar = np.eye(4, dtype=np.float64)   ### box2lidar是每个实例的外参
                box2lidar[:3, :3] = Rotation.from_quat(np.array(each_object['obj_rotation'])).as_matrix() # xyzw
                box2lidar[:3, 3] = np.array(each_object['obj_center_pos'])
                ### 1.对任何一个物体，先对应到激光雷达点云
                box2cam = lidar2camera @ box2lidar  ## 变换矩阵的乘法是从右向左应用
                # box2cam+cam2world(相机外参) box==object
                obj2world = cam2world @ box2cam
                obj2world_list.append(obj2world)
    # time.sleep(1000)
    return obj2world_list

def find_track_id_boxsize(track_id, data, frame_list):
    box_size_list = []
    for frame_id in range(len(frame_list)):
        frame_data = data['frames'][frame_id]
        object_detection_anns_info = frame_data.get('annotated_info', {}).get(
            '3d_city_object_detection_annotated_info', {}
        ).get('annotated_info', {}).get('3d_object_detection_info', {}).get(
            '3d_object_detection_anns_info', [])   
        ### frame_data中还是有很多track
        for i in range(len(object_detection_anns_info)):
            if object_detection_anns_info[i]['track_id'] == int(track_id):
                each_object = object_detection_anns_info[i]  
                box_size_list.append(each_object['size'])
    return box_size_list

def convert_ndarray_to_list(obj):
    """
    递归将数据结构中的所有ndarray转换为list
    """
    if isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj
    