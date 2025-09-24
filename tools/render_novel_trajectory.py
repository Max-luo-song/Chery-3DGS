from typing import List, Optional
from omegaconf import OmegaConf
import os
import time
import logging
import argparse
import numpy as np
import sys
# sys.path.append("/data4/gls/code/drivestudio")
import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import render_novel_views
from chery_tools.pc_generator import unproject_depth_to_pointcloud, remove_ground_points, save_pointcloud_pcd

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


@torch.no_grad()
def render_trajectory(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    cam_ids: Optional[List[int]] = None,
    downscales: Optional[List[int]] = None,
    render_rgb: bool = True,
    render_depth: bool = False,
    generate_lidar_pc: bool = False,
):
    trainer.set_eval()

    logger.info("Rendering novel views...")

    render_novel_cfg = cfg.render.get("render_novel", None)

    render_traj = dataset.get_novel_render_traj(
        traj_types=render_novel_cfg.traj_types,
        target_frames=render_novel_cfg.get("frames", dataset.frame_num),
    )
    
    render_keys = []
    if render_rgb:
        render_keys.append('rgbs')
    if render_depth:
        render_keys.append('depths')

    camera_data = dataset.pixel_source.load_specified_cameras(cam_ids, downscales)

    for traj_type, traj in render_traj.items():
        output_dir = f"{cfg.log_dir}/novel_traj/{traj_type}_step{step}"
        os.makedirs(output_dir, exist_ok=True)

        # traj 为单个相机（前向）的新轨迹, render_data 包含所有相机在该轨迹下的位姿等信息
        # FIXME(syc): CUDA OOM when downscale=2
        render_data = dataset.prepare_novel_view_render_data(traj, camera_data)

        # Render and save video
        video_output_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_output_dir, exist_ok=True)

        video_output_path = os.path.join(video_output_dir, f"{traj_type}.mp4")
        depth_maps = render_novel_views(
            trainer,
            render_data,
            camera_data,
            video_output_path,
            render_keys=render_keys,
            fps=render_novel_cfg.get("fps", cfg.render.fps),
        )

        logger.info(f"Saved novel view videos for trajectory type: {traj_type}") 

        # Release rendering data
        del render_data

        # Generate lidar point cloud files for each frame
        if depth_maps is not None and generate_lidar_pc:
            pc_output_dir = os.path.join(output_dir, "lidar_point_clouds")
            os.makedirs(pc_output_dir, exist_ok=True)

            # NOTE(syc): 这段代码只适用于奇瑞数据集
            pinhole_cam_ids = [id for id, cam in camera_data.items() if not cam.is_fisheye]
            intrinsics = {cam_id: camera_data[cam_id].intrinsics.cpu().numpy() for cam_id in pinhole_cam_ids}
            T_cam_to_lidars = {cam_id: camera_data[cam_id].cam_to_main_lidar for cam_id in pinhole_cam_ids}

            frame_id = 0
            for t in range(dataset.start_timestep, dataset.end_timestep):
                depths = depth_maps[frame_id]
                all_points = []
                for cam_id in pinhole_cam_ids:
                    points = unproject_depth_to_pointcloud(depths[cam_id], intrinsics[cam_id][frame_id], T_cam_to_lidars[cam_id])
                    if len(points) > 0:
                        all_points.append(points)

                if len(all_points) > 0:
                    # 合并所有相机的点云
                    all_points = np.vstack(all_points)
                    all_points = remove_ground_points(all_points, ground_height=-3)

                    # 保存为PCD文件
                    pcd_path = os.path.join(pc_output_dir, f"frame_{t:06d}.pcd")
                    save_pointcloud_pcd(all_points, pcd_path)
                    
                    logger.debug(f"Frame {t}: Merged {len(all_points)} points -> {pcd_path}")

                frame_id += 1


def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    args.enable_wandb = False

    if args.cam_ids is not None:
        args.cam_ids = sorted([int(x.strip()) for x in args.cam_ids.split(',')])
        # FIXME(syc): 无法针对不同相机设置不同的 downscale
        # downscales = [2] * len(args.cam_ids)
        downscales = [cfg.data.pixel_source.downscale_when_loading[0]] * len(args.cam_ids)
    
    if args.traj_types is not None:
        cfg.render.render_novel.traj_types = args.traj_types
    
    if args.fps is not None:
        cfg.render.render_novel.fps = args.fps
        
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)

    if args.save_catted_videos:  # 开启视频即拼接
        cfg.logging.save_seperate_video = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )

    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True,
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )

    render_trajectory(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        cam_ids=args.cam_ids,
        downscales=downscales,
        render_rgb=args.render_rgb,
        render_depth=args.render_depth,
        generate_lidar_pc=args.generate_lidar_pc
    )

    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Render novel trajectory for a single scene")
    # eval
    parser.add_argument(
        "--resume_from",
        default=None,
        help="path to checkpoint to resume from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--traj_types",
        default=None,
        nargs="+",
        type=str,
        help="Types of novel trajectories",
    )
    parser.add_argument(
        "--traj_dir",
        default=None,
        type=str,
        help="directory of self-defined trajectories (cam2world)",
    )
    parser.add_argument(
        "--fps",
        default=10,
        type=str,
        help="Frame per second of the rendered video",
    )
    parser.add_argument(
        "--cam_ids",
        default=None,
        type=str,
        help="Camera ID to render",
        required=True,
    )
    parser.add_argument(
        "--save_catted_videos",
        type=bool,
        default=False,
        help="visualize lidar on image",
    )

    parser.add_argument("--render_rgb", action="store_true", help='render rgb novel views')
    parser.add_argument("--render_depth", action="store_true", help='render depth novel views')
    parser.add_argument("--generate_lidar_pc", action="store_true", help='generate point cloud for each frame')

    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")

    # misc
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    main(args)
