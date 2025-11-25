import argparse
import os

import socket
import numpy as np
from cam.cam_inference_online import Renderer
from infer_utils import extract_lidar_extrinsics, load_transform_matrix, find_min_frame_txt
from datasets.qcraft.qcraft_utils import pose_to_transform_matrix


def recv_exact(conn, num_bytes):
    """循环接收直到 num_bytes 满足"""
    data = b""
    while len(data) < num_bytes:
        packet = conn.recv(num_bytes - len(data))
        if not packet:
            return None
        data += packet
    return data


def main(args):
    # -----------------------------
    # 初始化渲染器（只做一次）
    # -----------------------------
    renderer = Renderer(
        resume_from=args.resume_from,
        cam_ids=[0, 1, 2, 4, 5],         # 多相机
        downscales=[1, 1, 1, 1, 1],
        output_dir="./realtime_output"
    )
    lidar2ego_dict = extract_lidar_extrinsics(
        json_path=os.path.join(args.source_path, "data_frame_car_info.json")
    )
    lidar2ego = pose_to_transform_matrix(**lidar2ego_dict)
    lidar_pose_txt = find_min_frame_txt(os.path.join(args.source_path, "lidar_pose"))
    lidar_to_world_start = load_transform_matrix(txt_path=lidar_pose_txt)
    cam2lidar = load_transform_matrix(txt_path=os.path.join(args.source_path, "extrinsics", "0.txt"))
    
    inv_start = np.linalg.inv(lidar_to_world_start)

    print("[Socket] Waiting for connection...")

    # -----------------------------
    # Socket 服务端
    # -----------------------------
    # server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server.bind(("0.0.0.0", 6006))
    # server.listen(1)

    # conn, addr = server.accept()
    # print(f"[Socket] Connected from {addr}")

    # # -----------------------------
    # # 实时循环：收一帧 -> 渲染一帧
    # # -----------------------------
    # pose_size = 4 * 4 * 4   # float32 4x4

    while True:
        # raw = recv_exact(conn, pose_size)
        # if raw is None:
        #     print("[Socket] Client disconnected.")
        #     break


    #     raw =     [
    #   [
    #     -0.0003003308075243935,
    #     0.01738428679703722,
    #     0.9998488367618203,
    #     0.3501678758353342
    #   ],
    #   [
    #     -0.9998820195568772,
    #     -0.015360529289832272,
    #     -3.32685568167856e-05,
    #     0.002666504731999364
    #   ],
    #   [
    #     0.015357628992351628,
    #     -0.9997308841445756,
    #     0.017386849031343146,
    #     -0.09546387412984211
    #   ],
    #   [
    #     0.0,
    #     0.0,
    #     0.0,
    #     1.0
    #   ]
    # ]

        # test data
        raw = {        
        "x": 7987.086396549613,
        "y": -2964.653905917249,
        "z": -396.08292995686327,
        "yaw": 1.9627021134112075,
        "roll": 0,
        "pitch": 0}

        pose = pose_to_transform_matrix(**raw)
        # pose = np.array(raw, dtype=np.float32).reshape(4, 4)
        lidar2world = pose @ lidar2ego
        rel = inv_start @ lidar2world
        cam2world = rel @ cam2lidar

        print("[Frame] Received pose:\n", pose)

        # 渲染所有相机
        output_paths = renderer.render_single_frame(cam2world)

        print("[Frame] Rendered images:", output_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Render novel trajectory for a single scene")
    # eval
    parser.add_argument(
        "--resume_from",
        default="/home/workspace/scene_reconstruction_traj/output/qcraft_20251025_163358_QCOYSD504206_1595_1610/20251118_lidar+cam0_1_2_3_4_5_7_8_10/checkpoint_final.pth",
        help="path to checkpoint to resume from",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--source_path",
        default="/nas_thoru/oldbak/zyj/data/processed_new/training/20251025_163358_QCOYSD504206_1595_1610",
        help="data source path",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    main(args)