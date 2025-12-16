import numpy as np, argparse, os, shutil, json
import open3d as o3d
from typing import Dict, List
from scipy.spatial.transform import Rotation as R
from datasets.qcraft.qcraft_helpers import (
    OPENCV2DATASET,
    euler_to_transform_matrix,
)
from chery_tools.parse_lidar import parse_lidar_pcd_file

CAMERA_PERSPECTIVE = [
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110",  # 车内
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30",
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15",
    # "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_LEFT_H60",
    "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30",
    # "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_RIGHT_H60",
    "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30",
    "CAM_PBQ_REAR_RESET_OPTICAL_H50",
]

def gen_lidar_to_cam(ego2lidar, cam2ego) -> List:
    assert isinstance(cam2ego, list), "cam_exs shape error"
    assert isinstance(ego2lidar, list), "lidar_ex shape error"
    return np.linalg.inv(np.array(ego2lidar) @ np.array(cam2ego)).tolist()

def transform_lidar(lidar2ego, source_file, dst_file):
    ori_pcd = o3d.io.read_point_cloud(source_file)
    ori_pts = np.array(ori_pcd.points)
    xyz_ego_homo = np.hstack([ori_pts, np.ones((ori_pts.shape[0], 1))])  # [N, 4]
    xyz_lidar_homo = (np.linalg.inv(lidar2ego) @ xyz_ego_homo.T).T # [N, 4]
    ori_pcd.points = o3d.utility.Vector3dVector(xyz_lidar_homo[:, :3])
    o3d.io.write_point_cloud(dst_file, ori_pcd)


def organize_meta(args, all_time_stamps, lidar2ego):
    # create dirs
    for cam_pers in CAMERA_PERSPECTIVE:
        create_dirs(os.path.join(args.dst, "camera", cam_pers))
    create_dirs(os.path.join(args.dst, "lidar"))
    create_dirs(os.path.join(args.dst, "label"))

    # copy image and lidar into dst path
    for ts in all_time_stamps:
        for file in os.listdir(os.path.join(args.src, ts)):
            if file.endswith(".jpg"):
                assert len(file.split("-")) == 3
                cam_id = file.split("-")[1]
                src_file = os.path.join(args.src, ts, file)
                dst_path = os.path.join(args.dst, "camera", cam_id)
                if os.path.exists(dst_path):
                    shutil.copy(src_file, os.path.join(dst_path, ts + ".jpg"))
            elif file.endswith(".pcd"):
                src_file = os.path.join(args.src, ts, file)
                dst_path = os.path.join(args.dst, "lidar")
                if os.path.exists(dst_path):
                    transform_lidar(
                        lidar2ego, src_file, os.path.join(dst_path, ts + ".pcd")
                    )

def organize_calibs(args):
    # create dirs
    create_dirs(os.path.join(args.dst, "calib/camera"))

    raw_cam_calib_file_pth = os.path.join(args.src, "camera_params.json")
    raw_car_info_file_pth = os.path.join(args.src, "data_frame_car_info.json")

    if not os.path.exists(raw_cam_calib_file_pth):
        raise FileNotFoundError("cam params file not find")
    with open(raw_cam_calib_file_pth, encoding="utf-8") as f:
        cam_params = json.load(f)

    if not os.path.exists(raw_car_info_file_pth):
        raise FileNotFoundError("car info file not find")
    with open(raw_car_info_file_pth, encoding="utf-8") as f:
        car_infos = json.load(f)

    lidar_info = (
        car_infos.get("lidar_params", False)[0]
        .get("installation", False)
        .get("extrinsics", False)
    )

    lidar2ego = euler_to_transform_matrix(
        lidar_info["x"],
        lidar_info["y"],
        lidar_info["z"],
        lidar_info["yaw"],
        lidar_info["pitch"],
        lidar_info["roll"],
    )

    ego2lidar = np.linalg.inv(lidar2ego).tolist()

    # cal lidar to cam extrinsics
    lidar2cams = {}
    for cam_id in CAMERA_PERSPECTIVE:
        if cam_params.get(cam_id, False):
            caminfo = cam_params.get(cam_id, False).get(
                "camera_to_vehicle_extrinsics", False
            )
            cam2ego = euler_to_transform_matrix(
                caminfo["x"],
                caminfo["y"],
                caminfo["z"],
                caminfo["yaw"],
                caminfo["pitch"],
                caminfo["roll"],
            )
            cam2ego = cam2ego @ OPENCV2DATASET
            lidar2cams[cam_id] = gen_lidar_to_cam(ego2lidar, cam2ego.tolist())

    for i, cam_id in enumerate(CAMERA_PERSPECTIVE):
        if not lidar2cams.get(cam_id, False):
            continue
        dst_file_path = os.path.join(args.dst, "calib/camera", cam_id + ".json")
        intrinsic = [
            cam_params.get(cam_id, False).get("intrinsics", False).get("fx", False),
            0,
            cam_params.get(cam_id, False).get("intrinsics", False).get("cx", False),
            0,
            cam_params.get(cam_id, False).get("intrinsics", False).get("fy", False),
            cam_params.get(cam_id, False).get("intrinsics", False).get("cy", False),
            0,
            0,
            1,
        ]
        info = {
            "extrinsic": np.asarray(lidar2cams[cam_id]).flatten().tolist(),
            "intrinsic": intrinsic,
        }
        with open(dst_file_path, "w", encoding="utf-8") as f:
            json.dump(info, f)
            print(f"save {dst_file_path}")
    return lidar2ego


def create_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, mode=0o777, exist_ok=False)


def convert2sus(args):
    """
    sus标注目录:
     - camera
        - front
            - timestamp.jpg
        - left
            - timestamp.jpg
     - calib
        -camera
            - front.json
            - left.json
     - lidar
        - timestamp.pcd
     - label
        - timestamp.json
    """
    assert os.path.exists(args.src), "source clip path is not exists!"

    all_time_stamps_list = [
        f for f in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, f))
    ]

    lidar2ego = organize_calibs(args)
    organize_meta(args, all_time_stamps_list, lidar2ego)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("format qcraft data into sustechpoints")
    parser.add_argument("--src", help="qcraft clip path", type=str)
    parser.add_argument(
        "--dst", help="path to save organized sus-format data", type=str
    )

    args = parser.parse_args()
    convert2sus(args)

"""
python datasets/chery/chery_to_sustechpoints.py --src ~/Downloads/场景重建/qcraft_0702/20250702_133223_Q2517_60_75 --dst ~/Tools/SUSTechPOINTS/data/qcraft_test_1215
"""
