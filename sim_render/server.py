# server.py
# -*- coding: utf-8 -*-
import argparse
import struct
import socket
import threading
import Pose_pb2

import inference


class TCPServer:
    def __init__(self, host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False

    def start(self):
        """启动TCP服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True

            print(f"[Socket] Server listening on {self.host}:{self.port}")

            while self.running:
                client_socket, client_address = self.socket.accept()
                print(f"[Socket] New connection: {client_address}")

                t = threading.Thread(
                    target=self.handle_client_pb,
                    args=(client_socket, client_address)
                )
                t.daemon = True
                t.start()

        except Exception as e:
            print(f"[Socket] Server start error: {e}")

    def handle_client_pb(self, conn, addr):
        """接收 protobuf pose 并渲染"""
        try:
            while True:
                header = conn.recv(4)
                if not header:
                    return
                data_length = int.from_bytes(header, "big")

                data = b""
                while len(data) < data_length:
                    packet = conn.recv(data_length - len(data))
                    if not packet:
                        break
                    data += packet

                pose_msg = Pose_pb2.MainCarInfo()
                pose_msg.ParseFromString(data)

                print(f"[Socket] Received pose from {addr}: "
                      f"x={pose_msg.x}, y={pose_msg.y}, yaw={pose_msg.yaw}")

                # render cam images
                output = inference.cam_renderer_manager.render_from_pose(pose_msg)

                try:
                    for cam_id, image_path in output.items():
                        # 读取图片
                        with open(image_path, 'rb') as f:
                            image_data = f.read()

                        # 发送数据长度
                        conn.sendall(struct.pack('>I', len(image_data)))
                        # 发送图片数据
                        conn.sendall(image_data)

                        print(f"图片发送成功: {image_path}")

                except Exception as e:
                    print(f"发送失败: {e}")
                conn.send(b"OK")

        except Exception as e:
            print(f"[Socket] Error with {addr}: {e}")
        finally:
            conn.close()
            print(f"[Socket] Connection closed: {addr}")

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
        print("[Socket] Server stopped")


def main(args):
    # init renderer
    inference.init_renderer_cam(
        resume_from=args.resume_from,
        source_path=args.source_path
    )

    server = TCPServer()
    server.start()


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
