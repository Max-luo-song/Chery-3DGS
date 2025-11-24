import socket
import numpy as np
from cam.cam_inference_online import Renderer


def recv_exact(conn, num_bytes):
    """循环接收直到 num_bytes 满足"""
    data = b""
    while len(data) < num_bytes:
        packet = conn.recv(num_bytes - len(data))
        if not packet:
            return None
        data += packet
    return data


def main():
    # -----------------------------
    # 初始化渲染器（只做一次）
    # -----------------------------
    renderer = Renderer(
        resume_from="/home/workspace/scene_reconstruction_traj/output/qcraft_20251025_163358_QCOYSD504206_1595_1610/20251118_lidar+cam0_1_2_3_4_5_7_8_10/checkpoint_final.pth",
        cam_ids=[0, 1, 2],         # 多相机
        downscales=[1, 1, 1],
        output_dir="./realtime_output"
    )

    print("[Socket] Waiting for connection...")

    # -----------------------------
    # Socket 服务端
    # -----------------------------
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 6006))
    server.listen(1)

    conn, addr = server.accept()
    print(f"[Socket] Connected from {addr}")

    # -----------------------------
    # 实时循环：收一帧 -> 渲染一帧
    # -----------------------------
    pose_size = 4 * 4 * 4   # float32 4x4

    while True:
        raw = recv_exact(conn, pose_size)
        if raw is None:
            print("[Socket] Client disconnected.")
            break

        # 将 16 个 float32 解码为 4x4 矩阵
        pose = np.frombuffer(raw, dtype=np.float32).reshape(4, 4)

        print("[Frame] Received pose:\n", pose)

        # 渲染所有相机
        output_paths = renderer.render_single_frame(pose)

        print("[Frame] Rendered images:", output_paths)


if __name__ == "__main__":
    main()
