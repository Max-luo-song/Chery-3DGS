# renderer_manager.py
import os
import numpy as np
from cam.cam_inference_online import Renderer
from infer_utils import extract_lidar_extrinsics, load_transform_matrix, find_min_frame_txt
from datasets.qcraft.qcraft_utils import pose_to_transform_matrix


class CamRendererManager:
    def __init__(self, resume_from, source_path):
        print("[Init] Initializing renderer...")

        self.renderer = Renderer(
            resume_from=resume_from,
            cam_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            downscales=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            output_dir="./realtime_output"
        )

        # Preload the matrices required for trajectory coordinate transformation
        lidar2ego_dict = extract_lidar_extrinsics(
            json_path=os.path.join(source_path, "data_frame_car_info.json")
        )
        self.lidar2ego = pose_to_transform_matrix(**lidar2ego_dict)

        lidar_pose_txt = find_min_frame_txt(os.path.join(source_path, "lidar_pose"))
        lidar_to_world_start = load_transform_matrix(txt_path=lidar_pose_txt)
        self.cam2lidar = load_transform_matrix(txt_path=os.path.join(source_path, "extrinsics", "0.txt"))

        self.inv_start = np.linalg.inv(lidar_to_world_start)

        print("[Init] Renderer initialization done.")

    def render_from_pose(self, pose_msg):
        """
        pose_msg: 来自 protobuf 的 MainCarInfo，包括 x,y,z,yaw,pitch,roll
        """

        raw = {
            "x": pose_msg.x,
            "y": pose_msg.y,
            "z": pose_msg.z,
            "yaw": pose_msg.yaw,
            "roll": pose_msg.roll,
            "pitch": pose_msg.pitch,
        }

        pose = pose_to_transform_matrix(**raw)
        lidar2world = pose @ self.lidar2ego
        rel = self.inv_start @ lidar2world
        cam2world = rel @ self.cam2lidar

        print("[Render] Pose:\n", pose)

        output_paths = self.renderer.render_single_frame(cam2world)
        print("[Render] Output:", output_paths)


# initialized only once
cam_renderer_manager: CamRendererManager = None


def init_renderer_cam(resume_from, source_path):
    global cam_renderer_manager
    if cam_renderer_manager is None:
        cam_renderer_manager = CamRendererManager(resume_from, source_path)
    return cam_renderer_manager
