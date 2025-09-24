import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

from datasets.tools.multiprocess_utils import track_parallel_progress
from .qcraft_utils import (
    parse_lidar_pcd_file,
    project_points_to_image,
    draw_and_fill_box,
    filter_points_in_box,
    find_track_id_frame,
    find_track_id_obj2world,
    find_track_id_boxsize,
    convert_ndarray_to_list,
)

import cv2
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from scipy.spatial.transform import Rotation

QCRAFT_CLASSES = ["unknown", "Vehicle", "Pedestrian", "Sign", "Cyclist"]
# TODO(ziyu): consider all dynamic classes
QCRAFT_DYNAMIC_CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]
QCRAFT_HUMAN_CLASSES = ["Pedestrian", "Cyclist"]
QCRAFT_VEHICLE_CLASSES = ["Vehicle"]


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

        # NOTE(syc): 目前轻舟数据只有一个子目录
        subdir = os.listdir(self.load_dir)[0]
        self.load_dir = os.path.join(load_dir, subdir)

        self.save_dir = f"{save_dir}/{prefix}"
        self.workers = int(workers)

        self.num_pinhole_cameras = 13
        self.num_all_cameras = 13

        self.create_folder()

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

        if "images" in self.process_keys:
            self.save_image(clip_name)
            print(f"Processed images for {clip_name}")

        if "calib" in self.process_keys:
            self.save_calib(clip_name)
            print(f"Processed calib for {clip_name}")

        # if "lidar_velocities" in self.process_keys:
        #     self.save_lidar_velocities(clip_name)
        #     print(f"Processed lidar velocities for {clip_name}")

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
        clip_dir = os.path.join(self.load_dir, clip_name)

        sample_folders = [
            sample_folder
            for sample_folder in os.listdir(clip_dir)
            if os.path.isdir(os.path.join(clip_dir, sample_folder))
            and sample_folder.startswith("sample_")
        ]
        # 按帧的时间戳排序
        sample_folders.sort(key=lambda x: int(x.split("_")[-1]))

        for frame_index, sample_name in enumerate(sample_folders):
            sample_dir = os.path.join(clip_dir, sample_name)

            for img_name in os.listdir(sample_dir):
                if not img_name.startswith("camera"):
                    continue

                cam_idx = int(img_name.split("_")[0][6:])  # 0-10
                assert (
                    0 <= cam_idx < self.num_all_cameras
                ), f"Invalid camera index: {cam_idx}"

                # 针孔相机使用去畸变图像，鱼眼相机使用原始图像
                if cam_idx < self.num_pinhole_cameras and "_undist" not in img_name:
                    continue

                src_img_path = os.path.join(
                    sample_dir, img_name
                )  # camera0_1746752396953826_ae8a9af423443189bab574a9b3904cee_undist.jpg

                save_name = f"{frame_index:03}_{cam_idx}.jpg"  # 000_0, 000_1, ...
                save_img_path = os.path.join(
                    self.save_dir, clip_name, "images", save_name
                )
                image = Image.open(src_img_path)
                image.save(save_img_path)

    def save_calib(self, clip_name):
        # """解析并保存相机的外参（以LiDAR为参考系）和内参"""
        extrinsics = self._parse_extrinsics(clip_name)  # cam2lidar
        intrinsics = self._parse_intrinsics(clip_name)

        for cam_idx in range(self.num_all_cameras):
            np.savetxt(
                f"{self.save_dir}/{clip_name}/extrinsics/{cam_idx}.txt",
                extrinsics[cam_idx],
            )
            np.savetxt(
                f"{self.save_dir}/{clip_name}/intrinsics/{cam_idx}.txt",
                intrinsics[cam_idx],
            )
        pass

    def save_lidar(self, clip_name):
        # """
        # 将雷达数据从 pcd 格式转换到 bin 格式

        # 二进制文件中每个点包含 7 个 float32 字段：
        #     x y z intensity timestamp ring lidar_id
        # """

        # clip_dir = os.path.join(self.load_dir, clip_name)

        # lidar_slam_info_path = os.path.join(
        #     clip_dir, "static_obj/lidar_slam/lidar_slam_info.json"
        # )
        # if os.path.exists(lidar_slam_info_path):
        #     print("Found mc_pcds for lidar 0")

        #     with open(lidar_slam_info_path, "r") as f:
        #         lidar_slam_info = json.load(f)

        #     mc_lidar_paths = {
        #         info["frame_name"]: os.path.join(clip_dir, info["mc_lidar"])
        #         for info in lidar_slam_info["lidar0"]["mc_pcds"]
        #     }
        # else:
        #     print("No mc_pcds found for lidar 0")
        #     mc_lidar_paths = None

        # sample_folders = [
        #     sample_folder
        #     for sample_folder in os.listdir(clip_dir)
        #     if os.path.isdir(os.path.join(clip_dir, sample_folder))
        #     and sample_folder.startswith("sample_")
        # ]
        # # 按帧的时间戳排序
        # sample_folders.sort(key=lambda x: int(x.split("_")[-1]))

        # for frame_idx, sample_name in tqdm(enumerate(sample_folders)):
        #     sample_dir = os.path.join(clip_dir, sample_name)

        #     lidar_names = [
        #         lidar_name  # lidar0_1746752396900110_2e4bd97ab9d533658a28cfbe1581df57.pcd
        #         for lidar_name in os.listdir(sample_dir)
        #         if lidar_name.startswith("lidar")
        #         and not lidar_name.startswith("lidar5")  # 排除5号雷达（m2雷达）
        #     ]
        #     lidar_names = sorted(lidar_names, key=lambda x: int(x.split("_")[0][5:]))
        #     lidar_paths = [
        #         os.path.join(sample_dir, lidar_name) for lidar_name in lidar_names
        #     ]

        #     def save_lidar_data_as_bin(lidar_paths, lidar_type):
        #         point_clouds = [
        #             parse_lidar_pcd_file(lidar_path) for lidar_path in lidar_paths
        #         ]

        #         def append_lidar_id(pc, lidar_id):
        #             new_dtype = np.dtype(pc.dtype.descr + [("lidar_id", np.uint32)])
        #             new_pc = np.empty(pc.shape, dtype=new_dtype)

        #             for field in pc.dtype.names:
        #                 new_pc[field] = pc[field]

        #             new_pc["lidar_id"] = lidar_id
        #             return new_pc

        #         point_clouds = [
        #             append_lidar_id(pc, lidar_id)
        #             for lidar_id, pc in enumerate(point_clouds)
        #         ]

        #         # 点云数据合并
        #         point_cloud = np.concatenate(point_clouds, axis=0)

        #         # 提取所有字段并转换为 float32
        #         fields = ["x", "y", "z", "intensity", "timestamp", "ring", "lidar_id"]
        #         point_cloud = np.stack(
        #             [point_cloud[field] for field in fields], axis=1
        #         ).astype(np.float32)

        #         # 保存为二进制文件
        #         pc_path = f"{self.save_dir}/{clip_name}/{lidar_type}/{str(frame_idx).zfill(3)}.bin"
        #         point_cloud.astype(np.float32).tofile(pc_path)

        #     save_lidar_data_as_bin(lidar_paths, "lidar")
        pass

    def save_pose(self, clip_name):
        """保存每一帧的位姿"""

        # NOTE: 目前轻舟数据仿照的是奇瑞的数据预处理方式，因此保存的是 lidar pose 而非 ego pose
        # data_path = os.path.join(
        #     self.load_dir, f"{clip_name}/dynamic_obj/autolabel_10hz/{clip_name}.json"
        # )
        # with open(data_path, "r") as file:
        #     data = json.load(file)

        # for frame_idx, params in enumerate(data["frames"]):
        #     lidar_pose = params["lidar_pose"]
        #     lidar_pose = np.array(lidar_pose)
        #     np.savetxt(
        #         f"{self.save_dir}/{clip_name}/lidar_pose/{str(frame_idx).zfill(3)}.txt",
        #         lidar_pose,
        #     )
        pass

    # def save_lidar_velocities(self, clip_name):
    #     data_path = os.path.join(
    #         self.load_dir, f"{clip_name}/dynamic_obj/autolabel_10hz/{clip_name}.json"
    #     )
    #     with open(data_path, "r") as file:
    #         data = json.load(file)

    #     for frame_idx, params in enumerate(data["frames"]):
    #         lidar_velo = params["lidar_velo"]
    #         lidar_velo = np.array(lidar_velo)
    #         np.savetxt(
    #             f"{self.save_dir}/{clip_name}/lidar_velocities/{str(frame_idx).zfill(3)}.txt",
    #             lidar_velo,
    #         )

    def save_dynamic_mask(self, clip_name):
        # """需要雷达的 box 投影到图像平面上，获取 2D mask，包括all human vehicle三种"""

        # data_path = os.path.join(
        #     self.load_dir, f"{clip_name}/dynamic_obj/autolabel_10hz/{clip_name}.json"
        # )
        # with open(data_path, "r") as f:
        #     data = json.load(f)

        # # 读取 lidar2camera
        # extrinsics = self._parse_extrinsics(clip_name)  # cam2lidar
        # # extrinsics = self._parse_extrinsics(clip_name, pinhole_only=True)  # cam2lidar
        # lidar2cams = [np.linalg.inv(extrinsic) for extrinsic in extrinsics]

        # # 读取相机内参
        # intrinsics = [
        #     np.array(data["calibration"][f"camera{cam_idx}"]["intrinsic_scaled"])
        #     for cam_idx in range(self.num_all_cameras)
        #     # for cam_idx in range(self.num_pinhole_cameras)
        # ]

        # # 创建保存目录
        # categories = ["all", "human", "vehicle"]
        # for category in categories:
        #     mask_dir = f"{self.save_dir}/{clip_name}/dynamic_masks/{category}"
        #     if not os.path.exists(mask_dir):
        #         os.makedirs(mask_dir)

        # def process_object(object, category, masks, lidar2cams, intrinsics, img_shapes):
        #     """处理单个动态物体，生成掩码并更新图像"""
        #     l, w, h = object["size"]
        #     obj_rotation = np.array(object["obj_rotation"])
        #     obj_center_pos = np.array(object["obj_center_pos"])

        #     # 读取物体的位姿
        #     box2lidar = np.eye(4, dtype=np.float64)
        #     box2lidar[:3, :3] = Rotation.from_quat(obj_rotation).as_matrix()
        #     box2lidar[:3, 3] = obj_center_pos

        #     # # 1.对任何一个物体，先对应到激光雷达点云
        #     # mask_pointcloud = filter_points_in_box(pointcloud, obj_center_pos, size)
        #     # # 2.把激光点云向7个视角均投影得到 uv
        #     # # 3.七个视角最终全部投影得到结果
        #     for cam_idx, lidar2cam in enumerate(lidar2cams):
        #         box2cam = lidar2cam @ box2lidar
        #         qx, qy, qz, qw = Rotation.from_matrix(box2cam[:3, :3]).as_quat()
        #         cam_box = Box(
        #             center=box2cam[:3, 3],
        #             size=[w, l, h],
        #             orientation=Quaternion([qw, qx, qy, qz]),
        #             name=category,
        #         )
        #         corners = cam_box.corners().T.astype(np.float32)
        #         # print("corners:", corners)
        #         points2d = project_points_to_image(
        #             corners,
        #             intrinsics[cam_idx],
        #             img_shapes[cam_idx],
        #         )

        #         # 生成 mask
        #         masks[cam_idx] = draw_and_fill_box(masks[cam_idx], points2d)

        #     return masks

        # calib = data["calibration"]
        # img_shapes = [
        #     (
        #         calib[f"camera{cam_idx}"]["height"],
        #         calib[f"camera{cam_idx}"]["width"],
        #     )
        #     # for cam_idx in range(self.num_pinhole_cameras)
        #     for cam_idx in range(self.num_all_cameras)
        # ]

        # # 处理每一帧
        # for frame_idx, params in tqdm(
        #     enumerate(data["frames"]), total=len(data["frames"])
        # ):
        #     # 初始化掩码图像
        #     masks_vehicle = [
        #         np.zeros((sz[0], sz[1], 3), dtype=np.uint8) for sz in img_shapes
        #     ]
        #     masks_human = [
        #         np.zeros((sz[0], sz[1], 3), dtype=np.uint8) for sz in img_shapes
        #     ]
        #     masks_all = [
        #         np.zeros((sz[0], sz[1], 3), dtype=np.uint8) for sz in img_shapes
        #     ]

        #     # 处理物体检测信息
        #     object_anns = params["annotated_info"][
        #         "3d_city_object_detection_annotated_info"
        #     ]["annotated_info"]["3d_object_detection_info"][
        #         "3d_object_detection_anns_info"
        #     ]

        #     for obj in object_anns:
        #         category = obj["category"]

        #         # 类别对应关系
        #         # [Qcraft] -> [Waymo]
        #         # "person" -> "human"
        #         # 其余类别 -> "vehicle"

        #         if category == "person":
        #             masks_human = process_object(
        #                 obj,
        #                 category,
        #                 masks_human,
        #                 lidar2cams,
        #                 intrinsics,
        #                 img_shapes,
        #             )
        #         else:  # vehicle
        #             masks_vehicle = process_object(
        #                 obj,
        #                 category,
        #                 masks_vehicle,
        #                 lidar2cams,
        #                 intrinsics,
        #                 img_shapes,
        #             )

        #     for cam_idx in range(len(lidar2cams)):
        #         # 将 vehicle 和 human 掩码合并到 all 掩码中
        #         masks_all[cam_idx] = np.maximum(
        #             masks_all[cam_idx], masks_vehicle[cam_idx]
        #         )

        #     # 保存掩码图像
        #     for category, masks in zip(
        #         categories, [masks_all, masks_human, masks_vehicle]
        #     ):
        #         for cam_idx, mask in enumerate(masks):
        #             mask_gray = Image.fromarray(mask).convert("L")
        #             mask_path = os.path.join(
        #                 f"{self.save_dir}/{clip_name}/dynamic_masks/{category}",
        #                 f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.png",
        #             )
        #             mask_gray.save(mask_path)
        pass

    def save_objects(self, clip_name):
        # """
        # 生成instances相关的json文件
        # frame_instances是帧到实例的映射
        # instances_info是实例到属性的映射
        # """
        # frame_instances, instances_info = {}, {}

        # data_path = os.path.join(
        #     self.load_dir, f"{clip_name}/dynamic_obj/autolabel_10hz/{clip_name}.json"
        # )
        # with open(data_path, "r") as f:
        #     data = json.load(f)

        # """frame_instances.json"""
        # track_id_info = {}
        # for frame_index, frame_data in enumerate(data["frames"]):
        #     track_id_list = []
        #     object_detection_anns_info = (
        #         frame_data.get("annotated_info", {})
        #         .get("3d_city_object_detection_annotated_info", {})
        #         .get("annotated_info", {})
        #         .get("3d_object_detection_info", {})
        #         .get("3d_object_detection_anns_info", [])
        #     )
        #     for obj in object_detection_anns_info:
        #         track_id = obj.get("track_id")
        #         is_cyclist = obj.get("is_cyclist")
        #         # print("track_id:", track_id)
        #         # time.sleep(1000)
        #         category = obj.get("category")
        #         if track_id is not None and track_id not in track_id_list:
        #             track_id_list.append(track_id)
        #         ### 建立一个track_id和category的映射
        #         if track_id not in track_id_info:
        #             track_id_info[str(track_id)] = {
        #                 "category": category,
        #                 "is_cyclist": is_cyclist,
        #             }

        #     frame_instances[str(frame_index)] = track_id_list

        # """instances_info.json"""

        # print("Processing instances_info...")
        # instances_info = {}
        # ### result["实例编号"]["frame_annotations"]["frame_idx"] ["obj_to_world 4x4"] ["box_size 三维"]
        # # frame_idx可以反投影上面的result
        # # box_size是clip的size属性
        # # obj_to_world 每一个实例的旋转？？？  默认box就是obj

        # lidar2worlds = [np.array(frame["lidar_pose"]) for frame in data["frames"]]

        # for track_id, info in track_id_info.items():
        #     # print("type:", type(track_id))

        #     instances_info[track_id] = {}
        #     if "frame_annotations" not in instances_info[track_id]:
        #         instances_info[track_id]["frame_annotations"] = {}
        #     if "frame_idx" not in instances_info[track_id]["frame_annotations"]:
        #         instances_info[track_id]["frame_annotations"]["frame_idx"] = {}

        #     frame_list = find_track_id_frame(track_id, frame_instances)
        #     print("frame_list:", frame_list)
        #     instances_info[track_id]["frame_annotations"]["frame_idx"] = frame_list
        #     instances_info[track_id]["id"] = "track_" + track_id

        #     category = info["category"]
        #     is_cyclist = info["is_cyclist"]

        #     if category == "person":
        #         instances_info[track_id]["class_name"] = "Pedestrian"
        #     else:
        #         if is_cyclist == True:
        #             instances_info[track_id]["class_name"] = "Cyclist"
        #         else:
        #             instances_info[track_id]["class_name"] = "Vehicle"

        #     obj2world_list = find_track_id_obj2world(
        #         track_id,
        #         data,
        #         frame_list,
        #         lidar2worlds,
        #     )
        #     instances_info[track_id]["frame_annotations"][
        #         "obj_to_world"
        #     ] = obj2world_list
        #     box_size_list = find_track_id_boxsize(track_id, data, frame_list)
        #     instances_info[track_id]["frame_annotations"]["box_size"] = box_size_list

        # instances_info = convert_ndarray_to_list(instances_info)

        # return instances_info, frame_instances
        pass

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
            # if "lidar_velocities" in self.process_keys:
            #     os.makedirs(
            #         f"{self.save_dir}/{str(clip_name)}/lidar_velocities", exist_ok=True
            #     )
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

    # def _parse_extrinsics(self, clip_name, pinhole_only=False):
    #     extrinsics_dir = os.path.join(
    #         self.load_dir, f"{clip_name}/extrinsics/lidar2camera"
    #     )
    #     files = [
    #         "lidar2frontwide.yaml",
    #         "lidar2frontmain.yaml",
    #         "lidar2leftfront.yaml",
    #         "lidar2leftrear.yaml",
    #         "lidar2rightfront.yaml",
    #         "lidar2rightrear.yaml",
    #         "lidar2rearmain.yaml",
    #     ]
    #     extrinsics = []
    #     for file in files:
    #         with open(os.path.join(extrinsics_dir, file), "r", encoding="utf-8") as f:
    #             extrinsic = yaml.safe_load(f)

    #         lidar2cam = np.array(extrinsic["transform"])
    #         cam2lidar = np.linalg.inv(lidar2cam)

    #         extrinsics.append(cam2lidar)

    #     return extrinsics

    # def _parse_intrinsics(self, clip_name):
    #     data_path = os.path.join(
    #         self.load_dir, f"{clip_name}/dynamic_obj/autolabel_10hz/{clip_name}.json"
    #     )
    #     with open(data_path, "r") as f:
    #         data = json.load(f)

    #     intrinsics = []

    #     for cam_idx in range(self.num_all_cameras):
    #         params = data["calibration"][f"camera{cam_idx}"]

    #         # # 原始相机内参
    #         # intrinsic = params["intrinsic"]
    #         # fx = intrinsic[0][0]
    #         # cx = intrinsic[0][2]
    #         # fy = intrinsic[1][1]
    #         # cy = intrinsic[1][2]
    #         # intrinsic = [fx, fy, cx, cy]

    #         # 8 个畸变参数
    #         distcoeff = params["distcoeff"][0]

    #         # 去畸变后新的相机内参
    #         intrinsic_scaled = params["intrinsic_scaled"]
    #         fx_scaled = intrinsic_scaled[0][0]
    #         cx_scaled = intrinsic_scaled[0][2]
    #         fy_scaled = intrinsic_scaled[1][1]
    #         cy_scaled = intrinsic_scaled[1][2]
    #         intrinsic_scaled = [fx_scaled, fy_scaled, cx_scaled, cy_scaled]

    #         # values = intrinsic + distcoeff + intrinsic_scaled
    #         values = intrinsic_scaled + distcoeff
    #         intrinsics.append(values)

    #     return intrinsics
