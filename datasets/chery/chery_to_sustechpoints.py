import numpy as np, argparse, os, shutil, json
import open3d as o3d
from typing import Dict, List
from scipy.spatial.transform import Rotation as R

OPENCV2DATASET = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

CAMERA_PERSPECTIVE = [
    # "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
    # "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99",
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15",
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30",
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
    # "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110",  # 车内
    # "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_LEFT_H60",  # 车内
    # "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_RIGHT_H60",  # 车内
    # "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30",  # 分辨率小
    # "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30",  # 分辨率小
    # "CAM_PBQ_REAR_RESET_OPTICAL_H50",
    # "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99",
    # "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99",
]


def euler2transform(chery_extrinsics: Dict, type) -> List:
    """
    "camera_to_vehicle_extrinsics": {
            "pitch": -0.0007114634499885142,
            "roll": 0.020900780335068703,
            "x": 1.8500871658325195,
            "y": -0.035038694739341736,
            "yaw": 0.005182736553251743,
            "z": 1.4761027097702026
        }

    "lidar_extrinsics": {
     "calibration_time": "2025-06-14 10:57:50",
     "calibration_engineer": "cx",
     "calibration_run": "20250614_000005_Q2517",
     "calibration_mode": "kCalibrationIdle",
     "x": 1.7002451419830322,
     "y": 0.014630760066211224,
     "z": 1.5858668088912964,
     "yaw": 0.013984410092234612,
     "pitch": -0.0057430868037045,
     "roll": 0.0003618036862462759,
     "vehicle_name": ""
    },
    """
    assert isinstance(chery_extrinsics, dict)
    # 欧拉角（单位：弧度），顺序可选 'xyz', 'zyx' 等
    R_mat = R.from_euler(
        "ZYX",
        [
            chery_extrinsics.get("yaw"),
            chery_extrinsics.get("pitch"),
            chery_extrinsics.get("roll"),
        ],
        degrees=False,
    ).as_matrix()

    T = np.eye(4)
    if type == "camera":
        R_mat = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).T @ R_mat
    T[:3, :3] = R_mat
    T[:3, 3] = [
        chery_extrinsics.get("x"),
        chery_extrinsics.get("y"),
        chery_extrinsics.get("z"),
    ]
    return T.tolist()


def gen_lidar_to_cams(ego2lidar, cam_exs) -> List:
    assert isinstance(cam_exs[0], list), "cam_exs shape error"
    assert isinstance(ego2lidar, list), "lidar_ex shape error"
    return [(np.array(ego2lidar) @ (np.array(c2e) @ OPENCV2DATASET)).tolist() for c2e in cam_exs]

def transform_lidar(lidar2ego, source_file, dst_file):
    ori_pcd = o3d.io.read_point_cloud(source_file)
    ori_pts = np.asarray(ori_pcd.points)
    xyz_ego_homo = np.hstack([ori_pts, np.ones((ori_pts.shape[0], 1))])  # [N, 4]
    xyz_lidar_homo = (np.linalg.inv(lidar2ego) @ xyz_ego_homo.T).T  # [N, 4]
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
                # else:
                #     print(f"{dst_path} not exist, not include cam_id: {cam_id}")
            elif file.endswith(".pcd"):
                src_file = os.path.join(args.src, ts, file)
                dst_path = os.path.join(args.dst, "lidar")
                if os.path.exists(dst_path):
                    transform_lidar(lidar2ego, src_file, os.path.join(dst_path, ts + ".pcd"))
                    # shutil.copy(src_file, os.path.join(dst_path, ts + ".pcd"))


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

    lidar_ext = euler2transform(
        car_infos.get("lidar_params", False)[0]
        .get("installation", False)
        .get("extrinsics", False),
        "lidar",
    )
    
    ego2lidar = np.linalg.inv(lidar_ext).tolist()

    # cal lidar to cam extrinsics
    lidar2cams = gen_lidar_to_cams(
        ego2lidar,
        [
            euler2transform(
                cam_params.get(cam_id, False).get(
                    "camera_to_vehicle_extrinsics", False
                ),
                "camera",
            )
            for cam_id in CAMERA_PERSPECTIVE
        ],
    )

    for i, cam_id in enumerate(CAMERA_PERSPECTIVE):
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
            "extrinsic": np.asarray(lidar2cams[i]).flatten().tolist(),
            "intrinsic": intrinsic,
        }
        with open(dst_file_path, "w", encoding="utf-8") as f:
            json.dump(info, f)
    return lidar_ext

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
