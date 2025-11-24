import os
import time
import numpy as np
import torch

from omegaconf import OmegaConf
from utils.misc import import_str
from models.trainers import BasicTrainer
from datasets.driving_dataset_novel_view import DrivingDatasetNovelView
from models.video_utils import render_novel_views
from imageio import imwrite


class Renderer:
    """
    初始化模型、数据集，并暴露 render_single_frame() 给外部脚本实时调用
    """

    def __init__(self, resume_from: str, cam_ids, downscales, output_dir="./outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # ==== 加载配置 ====
        log_dir = os.path.dirname(resume_from)
        cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==== 数据集 ====
        self.dataset = DrivingDatasetNovelView(data_cfg=cfg.data)
        self.cam_ids = cam_ids
        self.downscales = downscales

        # 加载对应相机数据
        self.camera_data_dict = self.dataset.load_specified_cameras(
            self.cam_ids, self.downscales
        )

        # ==== 模型 ====
        self.trainer = import_str(cfg.trainer.type)(
            **cfg.trainer,
            num_timesteps=self.dataset.num_img_timesteps,
            model_config=cfg.model,
            num_full_images=len(self.dataset.train_indices + self.dataset.test_indices),
            test_set_indices=self.dataset.test_timesteps,
            scene_aabb=self.dataset.get_aabb().reshape(2, 3),
            device=device,
        )
        self.trainer.resume_from_checkpoint(
            ckpt_path=resume_from,
            load_only_model=True,
        )
        self.trainer.set_eval()

        print("[Renderer] Initialization completed.")

    @torch.no_grad()
    def render_single_frame(self, pose_cam2world: np.ndarray):
        """
        输入外部 4×4 位姿，对所有相机渲染一帧
        """
        render_results = {}

        # 单帧轨迹
        traj = {
            "cam2world": torch.from_numpy(pose_cam2world[None]).float().to(self.trainer.device),
            "frame_ids": [0],
        }

        for cam_id in self.cam_ids:
            cam_data = self.camera_data_dict[cam_id]

            # 数据准备
            render_data = self.dataset.prepare_novel_view_render_data(
                traj=traj,
                ref_cam_data=cam_data,      # 自看
                target_cam_data=cam_data,   # 自渲染
            )

            # 渲染
            results = render_novel_views(self.trainer, render_data, cam_data)

            rgb = results["rgbs"][0].cpu().numpy()
            rgb = (rgb * 255).astype(np.uint8)

            timestamp = time.time()
            out_path = os.path.join(self.output_dir, f"cam{cam_id}_{timestamp:.0f}.png")
            imwrite(out_path, rgb)

            render_results[cam_id] = out_path

        return render_results
