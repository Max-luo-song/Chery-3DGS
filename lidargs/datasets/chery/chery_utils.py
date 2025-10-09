import numpy as np
import cv2

from scipy.spatial.transform import Rotation


# FIXME: 好像是错的
def imu2ego():
    """
    imu在后轴中心，右前上 -> 车体坐标系在后轴中心地面点，前左上
    1) 右x，前y，上z -> 前x ，左y， 上z
    2) z平移到地面，E03轮半径: 0.349m
    """
    T = np.eye(4)
    r = Rotation.from_euler("z", -90, degrees=True)
    T[:3, :3] = r.as_matrix()
    T[2, 3] = 0.349  # E03轮半径
    return T


def parse_lidar_pcd_file(pcd_path):
    """
    头部信息示例:

    VERSION 0.7
    FIELDS x y z intensity timestamp ring
    SIZE 4 4 4 4 8 2
    TYPE F F F U F U
    COUNT 1 1 1 1 1 1
    WIDTH 182342
    HEIGHT 1
    VIEWPOINT 0.0 0.0 0.0 1.0 0.0 0.0 0.0
    POINTS 182342
    DATA binary
    """
    with open(pcd_path, "rb") as f:
        # 读取头部信息
        header = []
        while True:
            line = f.readline().decode("utf-8").strip()

            if line.startswith("DATA"):
                data_type = line.split()[1]
                if data_type != "binary":
                    raise ValueError("仅支持 binary 数据格式")
                break

            if line:
                header.append(line)

        # 解析头部信息
        def get_header_value(key):
            for h in header:
                if h.startswith(key):
                    return h.split()[1:]
            return None

        fields = get_header_value("FIELDS")
        sizes = [int(s) for s in get_header_value("SIZE")]
        types = get_header_value("TYPE")
        counts = [int(c) for c in get_header_value("COUNT")]
        points = int(get_header_value("POINTS")[0])

        # 构建 dtype
        dtype_list = []
        for i, field in enumerate(fields):
            count = counts[i]
            size = sizes[i]
            typ = types[i]

            if typ == "F":
                np_type = np.float32 if size == 4 else np.float64 if size == 8 else None
            elif typ == "U":
                np_type = (
                    np.uint8
                    if size == 1
                    else (
                        np.uint16
                        if size == 2
                        else (
                            np.uint32 if size == 4 else np.uint64 if size == 8 else None
                        )
                    )
                )
            elif typ == "I":
                np_type = (
                    np.int8
                    if size == 1
                    else (
                        np.int16
                        if size == 2
                        else np.int32 if size == 4 else np.int64 if size == 8 else None
                    )
                )
            else:
                raise ValueError(f"不支持的类型: {typ}")

            if np_type is None:
                raise ValueError(f"无效的大小 {size} 对于类型 {typ}")

            for j in range(count):
                field_name = f"{field}_{j}" if count > 1 else field
                dtype_list.append((field_name, np_type))

        dtype = np.dtype(dtype_list)
        pc = np.fromfile(f, dtype=dtype, count=points)

        return pc


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


def find_track_id_obj2world(track_id, data, frame_list, lidar2worlds):
    obj2world_list = []

    ### 每一个物体在每一个时间戳之内的obj2world
    for frame_id in range(len(frame_list)):
        frame_data = data["frames"][frame_id]
        object_detection_anns_info = (
            frame_data.get("annotated_info", {})
            .get("3d_city_object_detection_annotated_info", {})
            .get("annotated_info", {})
            .get("3d_object_detection_info", {})
            .get("3d_object_detection_anns_info", [])
        )
        ### frame_data中还是有很多track
        for i in range(len(object_detection_anns_info)):
            if object_detection_anns_info[i]["track_id"] == int(track_id):
                each_object = object_detection_anns_info[i]
                #### each_object是指定的track_id
                l, w, h = each_object["size"]  # ann_box is one of dynamic objs

                box2lidar = np.eye(4, dtype=np.float64)  ### box2lidar是每个实例的外参
                box2lidar[:3, :3] = Rotation.from_quat(
                    np.array(each_object["obj_rotation"])
                ).as_matrix()  # xyzw
                box2lidar[:3, 3] = np.array(each_object["obj_center_pos"])

                obj2world = lidar2worlds[frame_id] @ box2lidar
                obj2world_list.append(obj2world)

    return obj2world_list


def find_track_id_boxsize(track_id, data, frame_list):
    box_size_list = []
    for frame_id in range(len(frame_list)):
        frame_data = data["frames"][frame_id]
        object_detection_anns_info = (
            frame_data.get("annotated_info", {})
            .get("3d_city_object_detection_annotated_info", {})
            .get("annotated_info", {})
            .get("3d_object_detection_info", {})
            .get("3d_object_detection_anns_info", [])
        )
        ### frame_data中还是有很多track
        for i in range(len(object_detection_anns_info)):
            if object_detection_anns_info[i]["track_id"] == int(track_id):
                each_object = object_detection_anns_info[i]
                box_size_list.append(each_object["size"])
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
