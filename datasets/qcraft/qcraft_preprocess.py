import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass

from datasets.tools.multiprocess_utils import track_parallel_progress
from datasets.dataset_meta import DATASETS_CONFIG
from chery_tools.parse_lidar import parse_lidar_pcd_file
from .qcraft_utils import (
    euler_to_rotation_matrix,
    pose_to_transform_matrix,
    project_points_to_image,
    draw_and_fill_box,
    preprocess_lidar_point_cloud,
)

from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from scipy.spatial.transform import Rotation

QCRAFT_CLASSES = ["unknown", "Vehicle", "Pedestrian", "Sign", "Cyclist"]
# TODO(ziyu): consider all dynamic classes
QCRAFT_DYNAMIC_CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]
QCRAFT_HUMAN_CLASSES = ["Pedestrian", "Cyclist"]
QCRAFT_VEHICLE_CLASSES = ["Vehicle"]

QCRAFT_CAMERA_DICT = {  # 轻舟
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110": 0,  # 广角前视 FOV110
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60": 1,  # 广角前视 FOV60
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30": 2,  # 长焦前视 FOV30
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15": 3,  # 长焦前视 FOV15
    # "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_LEFT_H60": 4,  # 广角左前 FOV60
    "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99": 4,  # 左前 FOV99
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99": 5,  # 左后 FOV99
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30": 6,  # 左后 FOV30
    # "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_RIGHT_H60": 8,  # 广角右前 FOV60
    "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99": 7,  # 右前 FOV99
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99": 8,  # 右后 FOV99
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30": 9,  # 右后 FOV30
    "CAM_PBQ_REAR_RESET_OPTICAL_H50": 10,  # 后视 FOV50
}

OPENCV2DATASET = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Camera crop offsets for ego mask removal with proper intrinsic adjustment.
# Based on ACTUAL CROP RESULTS from ego_mask_crop_restore.py (2025-11-26):
# Camera 0 (front_wide_110, FOV110): Has invalid ego mask (all black), NO CROP NEEDED
#           Keep original 1024 width
# Camera 1 (front_wide_60, FOV60): Has invalid ego mask (all black), NO CROP NEEDED
#           Keep original 1024 width
# Camera 4: Ego on RIGHT (x=[772-1023]), cropped to LEFT x=[0-771] (772 pixels width)
#           Principle point: cx = 512 - 0 = 512 (stays at center of cropped region)
# Camera 7: Ego on LEFT (x=[0-258]), cropped to RIGHT x=[259-1023] (765 pixels width)
#           Principle point: cx = 512 - 259 = 253 (shifts left by 259 pixels)
QCRAFT_CAMERA_CROP_OFFSETS = {
    0: (0, 0, "down0"),
    1: (0, 0, "down1"),
    4: (0, 0, "right"),   # 裁剪右侧，可能需要额外信息
    # 7: (259, 0, "left"),  # 裁剪左侧
    7: (824, 0, "left"),  # 裁剪左侧
}

@dataclass
class ClipDataCache:
    clip_dir: str
    sample_names: List[str]
    ego_poses: List[np.ndarray]
    cam2egos: List[np.ndarray]
    intrinsics_matrix: List[np.ndarray]
    intrinsics_list: List[List[float]]
    lidar2ego: np.ndarray


class QcraftProcessor(object):
    """Process Qcraft dataset.

    Args:
        load_dir (str): Directory to load Qcraft raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        prefix,
        process_keys=[
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
            "objects",
            # "lidar_velocities",
        ],
        process_id_list=None,
        workers=64,
    ):
        # self.filter_no_label_zone_points = True

        # # Only data collected in specific locations will be converted
        # # If set None, this filter is disabled
        # # Available options: location_sf (main dataset)
        # self.selected_qcraft_locations = None
        # self.save_track_id = False

        self.process_id_list = process_id_list
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)

        self.load_dir = load_dir

        self.save_dir = f"{save_dir}/{prefix}"
        self.workers = int(workers)

        self.create_folder()

        self._cache: Dict[str, ClipDataCache] = {}

    def convert(self):
        """Convert action."""

        if self.process_id_list is None:
            print("No process_id_list is provided.")

        print("Start converting ...")
        track_parallel_progress(
            self.convert_one,
            self.process_id_list,
            self.workers,
        )
        print("\nFinished ...")

    def convert_one(self, clip_name):
        """Convert action for single file."""

        self._cache[clip_name] = {}

        # NOTE(syc): 轻舟数据下有一个子目录
        original_clip_dir = os.path.join(self.load_dir, clip_name)
        subdirs = os.listdir(original_clip_dir)
        clip_dir = os.path.join(original_clip_dir, subdirs[0])

        sample_names = self._read_sample_names(clip_dir)
        ego_poses = self._read_ego_poses(clip_dir, sample_names) # ego2world(throu)

        camera_params = self._read_camera_params(clip_dir)
        cam2egos = self._parse_extrinsics(camera_params) # 外参 cam2egos
        intrinsics_matrix, intrinsics_list = self._parse_intrinsics(camera_params)

        lidar2ego = self._read_lidar2ego(clip_dir)

        self._cache[clip_name] = ClipDataCache(
            clip_dir=clip_dir,
            sample_names=sample_names,
            ego_poses=ego_poses,
            cam2egos=cam2egos,
            intrinsics_matrix=intrinsics_matrix,
            intrinsics_list=intrinsics_list,
            lidar2ego=lidar2ego,
        )

        if "images" in self.process_keys:
            self.save_image(clip_name)
            print(f"Processed images for {clip_name}")

        if "calib" in self.process_keys:
            self.save_calib(clip_name)
            print(f"Processed calib for {clip_name}")

        if "lidar" in self.process_keys:
            self.save_lidar(clip_name)
            print(f"Processed lidar for {clip_name}")

        if "pose" in self.process_keys:
            self.save_pose(clip_name)
            print(f"Processed lidar poses for {clip_name}")

        if "dynamic_masks" in self.process_keys:
            self.save_dynamic_mask(clip_name)
            print(f"Processed dynamic masks for scene {clip_name}")

        if "objects" in self.process_keys:
            instances_info, frame_instances = self.save_objects(clip_name)
            print(f"Processed objects for scene {clip_name}")

            # Save instances info and frame instances
            object_info_dir = f"{self.save_dir}/{str(clip_name).zfill(3)}/instances"
            with open(f"{object_info_dir}/instances_info.json", "w") as fp:
                json.dump(instances_info, fp, indent=4)
            with open(f"{object_info_dir}/frame_instances.json", "w") as fp:
                json.dump(frame_instances, fp, indent=4)

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_id_list)

    def save_image(self, clip_name):
        """保存经过畸变校正的相机图像为 jpg 格式"""
        cache = self._cache[clip_name]

        img_paths = []
        save_paths = []

        for frame_idx, sample_name in enumerate(cache.sample_names):
            sample_dir = os.path.join(cache.clip_dir, sample_name)
            img_names = [
                img_name  # img_name: 20250702_133223_Q2517-CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99-1751434420.0459.jpg
                for img_name in os.listdir(sample_dir)
                if img_name.endswith(".jpg")
            ]

            for img_name in img_names:
                # 求出相机编号
                cam_idx = None
                for cam_name in QCRAFT_CAMERA_DICT:
                    if cam_name in img_name:
                        cam_idx = QCRAFT_CAMERA_DICT[cam_name]
                        break
                assert cam_idx is not None, f"Unknown camera in {img_name}"

                img_path = os.path.join(sample_dir, img_name)
                img_paths.append(img_path)

                save_name = f"{frame_idx:03}_{cam_idx}.jpg"  # 000_0, 000_1, ...
                save_path = os.path.join(self.save_dir, clip_name, "images", save_name)
                save_paths.append(save_path)

        for img_path, save_path in tqdm(
            zip(img_paths, save_paths), total=len(img_paths)
        ):
            Image.open(img_path).save(save_path)

    def save_calib(self, clip_name):
        cache = self._cache[clip_name]

        cam2lidars = [
            np.linalg.inv(cache.lidar2ego) @ cam2ego for cam2ego in cache.cam2egos
        ]
        intrinsics = cache.intrinsics_list

        for cam_idx, (cam2lidar, intrinsic) in tqdm(
            enumerate(zip(cam2lidars, intrinsics)), total=len(cam2lidars)
        ):
            np.savetxt(
                f"{self.save_dir}/{clip_name}/extrinsics/{cam_idx}.txt",
                cam2lidar,
            )
            np.savetxt(
                f"{self.save_dir}/{clip_name}/intrinsics/{cam_idx}.txt",
                intrinsic,
            )

    def save_lidar(self, clip_name):
        """
        将雷达数据从 pcd 格式转换到 bin 格式

        二进制文件中包含 5 个 float32 字段：
            x y z intensity lidar_id
        """
        cache = self._cache[clip_name]

        for frame_idx, sample_name in tqdm(
            enumerate(cache.sample_names), total=len(cache.sample_names)
        ):
            sample_dir = os.path.join(cache.clip_dir, sample_name)

            lidar_names = [
                lidar_name  # 20250702_133223_Q2517-LDR_FRONT-1751434420.2514-ego.pcd
                for lidar_name in os.listdir(sample_dir)
                if lidar_name.endswith(".pcd") and "LDR_FRONT" in lidar_name
            ]
            lidar_paths = [
                os.path.join(sample_dir, lidar_name) for lidar_name in lidar_names
            ]
            lidar_ids = [0]

            point_cloud_list = [
                parse_lidar_pcd_file(lidar_path) for lidar_path in lidar_paths
            ]  # x y z intensity
            point_cloud_list = [
                preprocess_lidar_point_cloud(pc, cache.lidar2ego, lidar_id)
                for lidar_id, pc in zip(lidar_ids, point_cloud_list)
            ]  # x y z intensity lidar_id
            point_cloud = np.concatenate(point_cloud_list, axis=0)

            # 保存为二进制文件
            bin_path = (
                f"{self.save_dir}/{clip_name}/lidar/{str(frame_idx).zfill(3)}.bin"
            )
            point_cloud.tofile(bin_path)

    def save_pose(self, clip_name):
        """保存每一帧的位姿"""
        # NOTE: 目前轻舟数据仿照的是奇瑞的数据预处理方式，因此保存的是 lidar pose 而非 ego pose

        cache = self._cache[clip_name]

        lidar2worlds = [ego2world @ cache.lidar2ego for ego2world in cache.ego_poses]
        for frame_idx, lidar2world in tqdm(
            enumerate(lidar2worlds), total=len(lidar2worlds)
        ):
            np.savetxt(
                f"{self.save_dir}/{clip_name}/lidar_pose/{str(frame_idx).zfill(3)}.txt",
                lidar2world,
            )

    def _get_obj_pose_and_size(
        self, obj_data
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        返回 obj2lidar, (l, w, h)
        """
        rotation = obj_data["psr"]["rotation"]
        rotation = euler_to_rotation_matrix(rotation["z"], rotation["y"], rotation["x"])

        position = obj_data["psr"]["position"]
        position = np.array([position["x"], position["y"], position["z"]])

        obj2ego = np.eye(4, dtype=np.float64)
        obj2ego[:3, :3] = rotation
        obj2ego[:3, 3] = position

        scale = obj_data["psr"]["scale"]
        l, w, h = scale["x"], scale["y"], scale["z"]

        return obj2ego, (l, w, h)

    def _generate_obj_masks(self, obj_data, masks, ego2cams, intrinsics):
        """
        处理单个动态物体，生成掩码图像
        """
        obj2ego, (l, w, h) = self._get_obj_pose_and_size(obj_data)

        for cam_idx, ego2cam in enumerate(ego2cams):
            obj2cam = ego2cam @ obj2ego
            box = Box(
                center=obj2cam[:3, 3],
                size=[w, l, h],
                orientation=Quaternion(matrix=obj2cam[:3, :3]),
            )
            corners_cam = box.corners().T.astype(np.float32)
            corners_2d = project_points_to_image(
                corners_cam,
                intrinsics[cam_idx],
            )
            masks[cam_idx] = draw_and_fill_box(masks[cam_idx], corners_2d)

        return masks

    def save_dynamic_mask(self, clip_name):
        """
        将 box 投影到图像平面上，获取 2D mask，包括 all human vehicle 三种
        """
        cache = self._cache[clip_name]

        # 创建保存目录
        categories = ["all", "human", "vehicle"]
        for obj_type in categories:
            mask_dir = f"{self.save_dir}/{clip_name}/dynamic_masks/{obj_type}"
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)

        img_shapes = [
            config["original_size"] for _, config in DATASETS_CONFIG["qcraft"].items()
        ]

        ego2cams = [np.linalg.inv(m) for m in cache.cam2egos]

        # 处理每一帧
        for frame_idx, sample_name in tqdm(
            enumerate(cache.sample_names), total=len(cache.sample_names)
        ):
            label_path = os.path.join(cache.clip_dir, "label", f"{sample_name}.json")
            with open(label_path, "r") as f:
                labels = json.load(f)

            # 初始化掩码图像
            masks_vehicle = [
                np.zeros((sz[0], sz[1], 3), dtype=np.uint8) for sz in img_shapes
            ]
            masks_human = [
                np.zeros((sz[0], sz[1], 3), dtype=np.uint8) for sz in img_shapes
            ]
            masks_all = [
                np.zeros((sz[0], sz[1], 3), dtype=np.uint8) for sz in img_shapes
            ]

            for obj_data in labels:
                obj_type = obj_data["obj_type"]
                if obj_type == "Person":
                    masks_human = self._generate_obj_masks(
                        obj_data,
                        masks_human,
                        ego2cams,
                        cache.intrinsics_matrix,
                    )
                else:
                    masks_vehicle = self._generate_obj_masks(
                        obj_data,
                        masks_vehicle,
                        ego2cams,
                        cache.intrinsics_matrix,
                    )

            for cam_idx in range(len(ego2cams)):
                # 将 vehicle 和 human 掩码合并到 all 掩码中
                masks_all[cam_idx] = np.maximum(
                    masks_all[cam_idx], masks_vehicle[cam_idx]
                )

            # 保存掩码图像
            for obj_type, masks in zip(
                categories, [masks_all, masks_human, masks_vehicle]
            ):
                for cam_idx, mask in enumerate(masks):
                    mask_gray = Image.fromarray(mask).convert("L")
                    mask_path = os.path.join(
                        f"{self.save_dir}/{clip_name}/dynamic_masks/{obj_type}",
                        f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.png",
                    )
                    mask_gray.save(mask_path)

    def save_objects(self, clip_name):
        """
        生成instances相关的json文件
        frame_instances是帧到实例的映射
        instances_info是实例到属性的映射
        """
        cache = self._cache[clip_name]

        instances_info, frame_instances = {}, {}

        for frame_idx, sample_name in tqdm(
            enumerate(cache.sample_names), total=len(cache.sample_names)
        ):
            labels_path = os.path.join(cache.clip_dir, "label", f"{sample_name}.json")
            with open(labels_path, "r") as f:
                obj_data_list = json.load(f)

            """
            instances_info = {
                "0": # simplified instance id
                    {
                        "id": str,
                        "class_name": str,
                        "frame_annotations": {
                            "frame_idx": List,
                            "obj_to_world": List,
                            "box_size": List,
                    },
                ...
            }
            """
            for obj_data in obj_data_list:
                obj_id: str = obj_data["obj_id"]

                obj2ego, box_size = self._get_obj_pose_and_size(obj_data)
                obj2world = cache.ego_poses[frame_idx] @ obj2ego

                if obj_id not in instances_info:
                    obj_type = obj_data["obj_type"]
                    if obj_type == "Person":
                        class_name = "Pedestrian"
                    elif obj_type == "Motorcycle":
                        class_name = "Cyclist"
                    else:
                        class_name = "Vehicle"

                    instances_info[obj_id] = {
                        "class_name": class_name,
                        "id": obj_id,
                        "frame_annotations": {
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                        },
                    }

                instances_info[obj_id]["frame_annotations"]["frame_idx"].append(
                    frame_idx
                )
                instances_info[obj_id]["frame_annotations"]["obj_to_world"].append(
                    obj2world.tolist()
                )
                instances_info[obj_id]["frame_annotations"]["box_size"].append(box_size)

            """
            frame_instances = {
                "0": # frame idx
                    List[int] # list of simplified instance ids
                ...
            }
            """
            frame_instances[str(frame_idx)] = [
                int(obj_data["obj_id"]) for obj_data in obj_data_list
            ]

        return instances_info, frame_instances

    def create_folder(self):
        """Create folder for data preprocessing."""

        for clip_name in self.process_id_list:
            if "images" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(clip_name)}/images", exist_ok=True)
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/sky_masks", exist_ok=True
                )
            if "calib" in self.process_keys:
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/extrinsics", exist_ok=True
                )
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/intrinsics", exist_ok=True
                )
            if "pose" in self.process_keys:
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/lidar_pose", exist_ok=True
                )
            if "lidar" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(clip_name)}/lidar", exist_ok=True)
            if "dynamic_masks" in self.process_keys:
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/dynamic_masks", exist_ok=True
                )
            if "objects" in self.process_keys:
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/instances", exist_ok=True
                )

    def _read_lidar2ego(self, clip_dir):
        data_frame_car_info_path = os.path.join(clip_dir, "data_frame_car_info.json")
        with open(data_frame_car_info_path, "r") as f:
            data_frame_car_info = json.load(f)

        lidar2ego_raw = data_frame_car_info["lidar_params"][0]["installation"][
            "extrinsics"
        ]
        lidar2ego = pose_to_transform_matrix(
            lidar2ego_raw["x"],
            lidar2ego_raw["y"],
            lidar2ego_raw["z"],
            lidar2ego_raw["yaw"],
            lidar2ego_raw["pitch"],
            lidar2ego_raw["roll"],
        )
        return lidar2ego

    def _read_ego_pose_from_pbtxt(self, frame_data_path):
        main_timestamp = None

        image_infos = []
        current_image_info = None

        with open(frame_data_path, "r", encoding="utf-8") as f:
            frame_data = f.read()

        frame_data = frame_data.strip().split("\n")

        for line in frame_data:
            line = line.strip()
            if not line:
                continue

            if line.startswith("main_timestamp") and main_timestamp is None:
                main_timestamp = float(line.split(":")[1].strip())
            elif line.startswith("image_infos"):
                current_image_info = {
                    "timestamp": None,
                    "vehicle_pose": None,
                }
            elif current_image_info:
                if line.startswith("timestamp:"):
                    current_image_info["timestamp"] = float(line.split(":")[1].strip())
                elif line.startswith("vehicle_pose"):
                    current_image_info["vehicle_pose"] = {}
                elif current_image_info["vehicle_pose"] is not None:
                    if line.startswith("}"):
                        image_infos.append(current_image_info)
                        current_image_info = None
                    else:
                        key, value = line.split(":")
                        key = key.strip()
                        value = float(value.strip())
                        current_image_info["vehicle_pose"][key] = value

        for info in image_infos:
            timestamp = info["timestamp"]
            if timestamp == main_timestamp:
                pose = info["vehicle_pose"]
                pose_matrix = pose_to_transform_matrix(
                    pose["x"],
                    pose["y"],
                    pose["z"],
                    pose["yaw"],
                    pose["pitch"],
                    pose["roll"],
                )
                return pose_matrix

        # HACK(syc): 可能没有匹配的 timestamp
        raise ValueError("No matching vehicle_pose found for main_timestamp")

    def _read_ego_poses(self, clip_dir, sample_names):
        ego_poses = []
        for sample_name in sample_names:
            sample_dir = os.path.join(clip_dir, sample_name)
            frame_data_path = os.path.join(sample_dir, "data_frame.pb.txt")
            ego_pose = self._read_ego_pose_from_pbtxt(frame_data_path)
            ego_poses.append(ego_pose)
        return ego_poses

    def _read_sample_names(self, clip_dir):
        data_frame_seq_path = os.path.join(clip_dir, "data_frame_seq.json")
        with open(data_frame_seq_path, "r") as f:
            data_frame_seq = json.load(f)
        sample_names = [
            item["data_frame_path"] for item in data_frame_seq["data_frame_seq_items"]
        ]
        return sample_names

    def _read_camera_params(self, clip_dir):
        camera_params_path = os.path.join(clip_dir, "camera_params.json")
        with open(camera_params_path, "r") as f:
            camera_params = json.load(f)
        return camera_params

    def _parse_extrinsics(self, camera_params):
        extrinsics = []
        for cam_name, _ in QCRAFT_CAMERA_DICT.items():
            cam2ego = camera_params[cam_name]["camera_to_vehicle_extrinsics"]
            cam2ego = pose_to_transform_matrix(
                cam2ego["x"],
                cam2ego["y"],
                cam2ego["z"],
                cam2ego["yaw"],
                cam2ego["pitch"],
                cam2ego["roll"],
            )
            cam2ego = cam2ego @ OPENCV2DATASET
            extrinsics.append(cam2ego)

        return extrinsics

    def _parse_intrinsics(self, camera_params):
        intrinsics_matrix, intrinsics_list = [], []
        for cam_name, _ in QCRAFT_CAMERA_DICT.items():
            intrinsic_raw = camera_params[cam_name]["intrinsics"]

            fx = intrinsic_raw["fx"]
            fy = intrinsic_raw["fy"]
            cx = intrinsic_raw["cx"]
            cy = intrinsic_raw["cy"]

            # 8 个畸变参数
            k1 = intrinsic_raw["k1"]
            k2 = intrinsic_raw["k2"]
            p1 = intrinsic_raw["p1"]
            p2 = intrinsic_raw["p2"]
            k3 = intrinsic_raw["k3"]
            k4 = intrinsic_raw["k4"]
            k5 = intrinsic_raw["k5"]
            k6 = intrinsic_raw["k6"]

            # NOTE(gls): 对裁剪图片的内参进行调整，写入内参txt的是调整后的内参
            # Apply crop offset adjustment for ego mask cropping
            # Camera 4 and 7 have been cropped to remove ego vehicle body regions
            cam_idx = QCRAFT_CAMERA_DICT[cam_name]
            if cam_idx in QCRAFT_CAMERA_CROP_OFFSETS:
                crop_offset_x, crop_offset_y, crop_direction = QCRAFT_CAMERA_CROP_OFFSETS[cam_idx]
                if crop_direction == "left":
                    # 裁剪左侧：调整偏移
                    cx = cx - crop_offset_x
                    cy = cy - crop_offset_y
                elif crop_direction == "right":
                    # 裁剪右侧：按比例缩放cx
                    cx = cx - crop_offset_x
                    cy = cy - crop_offset_y
                    # original_width = 1024
                    # new_width = 772 # 假设crop_offset_x表示右侧裁剪的像素数
                    # cx = cx * (new_width / original_width)
                elif crop_direction == "down0":
                    cx = cx - crop_offset_x
                    cy = cy - crop_offset_y
                    # original_height = 512  # 需要原始宽度信息
                    # new_height = 342 # 假设crop_offset_x表示右侧裁剪的像素数
                    # cy = cy * (new_height / original_height)
                elif crop_direction == "down1":
                    cx = cx - crop_offset_x
                    cy = cy - crop_offset_y
                    # 裁剪右侧：按比例缩放cx
                    # original_height = 512  # 需要原始宽度信息
                    # new_height = 472 # 假设crop_offset_x表示右侧裁剪的像素数
                    # cy = cy * (new_height / original_height)
                print(f"Camera {cam_idx}: Adjusted principal point by offset ({crop_offset_x}, {crop_offset_y})")
                print(f"  Original: ({intrinsic_raw['cx']}, {intrinsic_raw['cy']}) -> Adjusted: ({cx}, {cy})")
            matrix_values = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            list_values = [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]

            intrinsics_matrix.append(matrix_values)
            intrinsics_list.append(list_values)

        return intrinsics_matrix, intrinsics_list
