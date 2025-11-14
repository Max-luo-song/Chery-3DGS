import os
import torch
import numpy as np
import sys
import subprocess

cmd = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
result = (
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split("\n")
)
os.environ["CUDA_VISIBLE_DEVICES"] = str(
    np.argmin([int(x.split()[2]) for x in result[:-1]])
)

os.system("echo $CUDA_VISIBLE_DEVICES")

sys.path.append("/scene_reconstruction/lidargs/")
from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel, renderComposite
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
from scene import Scene

import cv2
from utils.lidar_utils import PointsMeter, pano_to_lidar_with_intensities, filter_pcd
from utils.loss_utils import l1_loss
from utils.image_utils import psnr
from scene.cameras import Camera
import open3d as o3d

# from utils.obj_utils import get_obj_type, loadStaticObj
from utils.data_partition_utils import getBlockInfo
import math

cpu_count = os.cpu_count()
torch.set_num_threads(cpu_count)

from typing import NamedTuple


class GaussianView(NamedTuple):
    gaussians: GaussianModel
    scene: Scene
    time_poses: dict


class TimePose(NamedTuple):
    camera_pose: np.array
    view: Camera
    valid_mask: np.array
    gt_mask: torch.Tensor


class ValidModeInfo(NamedTuple):
    model_id: int
    model_pose: np.array
    model_view: Camera
    model_gaussians: GaussianModel
    need_train: bool


class EditObjInfo(NamedTuple):
    delete_obj_ids: list
    obj_id_offset_pairs: list
    add_id_path_pairs: list


def render_set(
    gt_dynamic_model,
    dataset,
    name,
    iteration,
    valid_timestamp_model,
    model_id_scene_info,
    views,
    pipeline,
    background,
    insert_objs,
):
    path_name = dataset.model_path.split("/")
    render_path = os.path.join(dataset.model_path, "renders")
    gt_path = os.path.join(dataset.model_path, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    name_list = []
    per_view_dict = {}
    t_list = []  # 存储每帧渲染时间
    total_frames = len(views)  # 总渲染帧数

    # 记录总渲染开始时间
    total_render_start = time.time()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        render_timestamp = view.image_name
        world_to_camera_pose = np.eye(4)
        world_to_camera_pose[:3, :3] = np.transpose(view.R)
        world_to_camera_pose[:3, 3] = view.T
        camera_to_world_pose = np.linalg.inv(world_to_camera_pose)

        valid_model_info = []
        for model_id in valid_timestamp_model[render_timestamp]:
            time_pose = model_id_scene_info[model_id].time_poses[render_timestamp]
            model_to_camera_pose = time_pose.camera_pose
            object_view = time_pose.view

            model_gaussian = model_id_scene_info[model_id].gaussians
            model_gaussian.eval()
            model_to_world = camera_to_world_pose @ model_to_camera_pose
            valid_model_info.append(
                ValidModeInfo(
                    model_id=model_id,
                    model_pose=model_to_world,
                    model_view=object_view,
                    model_gaussians=model_gaussian,
                    need_train=False,
                )
            )

        torch.cuda.synchronize()
        t0 = time.time()
        render_pkg = renderComposite(
            view,
            background,
            pipeline,
            valid_model_info,
            dataset.max_depth,
            insert_objs=insert_objs,
            retain_grad=False,
        )
        torch.cuda.synchronize()
        t1 = time.time()
        # 输出当前进程占用的显存（单位：MB）
        mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(
            f"[GPU] 当前帧显存占用: allocated={mem_allocated:.2f} MB, reserved={mem_reserved:.2f} MB"
        )
        t_list.append(t1 - t0)  # 记录单帧渲染时间

        rendering = render_pkg["render"]
        render_intensity = rendering[0:1, ...]
        depth = render_pkg["depth"]

        gt = view.original_image.cuda()
        ray_drop = gt[0:1, ...]
        gt_intensity = (gt[1:2, ...] * ray_drop).detach().cpu().numpy()
        gt_depth = (gt[2:3, ...] * ray_drop).detach().cpu().numpy()
        render_raydrop = rendering[1:2, ...]
        render_raydrop_mask = torch.where(render_raydrop > 0.5, 1, 0)

        render_intensity = render_intensity * render_raydrop_mask
        depth = depth * render_raydrop_mask

        if True:
            depth_distortion_aware = render_pkg["mid_depth_diff"]
            depth_distortion_aware = torch.where(depth_distortion_aware < 0.3, 1, 0)
            render_intensity = render_intensity * depth_distortion_aware
            depth = depth * depth_distortion_aware

        depth_numpy = depth.detach().cpu().numpy()
        intensity_numpy = render_intensity.detach().cpu().numpy()

        point_with_intensity = pano_to_lidar_with_intensities(
            depth_numpy[0, :, :],
            intensity_numpy[0],
            lidar_K=None,
            beam_inclinations=view.beam_inclinations.detach().cpu().numpy(),
        )
        gt_point_with_intensity = pano_to_lidar_with_intensities(
            gt_depth[0, :, :],
            gt_intensity[0],
            lidar_K=None,
            beam_inclinations=view.beam_inclinations.detach().cpu().numpy(),
        )

        if False:  # 密度滤波
            make_raydrop = filter_pcd(point_with_intensity[:, :3])
            point_with_intensity = point_with_intensity[make_raydrop]

        if False:  # 转到baselidar系
            sensor2baselidar = gt_dynamic_model.get_sensor2baselidar(dataset.sensorid)
            points = point_with_intensity[:, :3]
            points = (
                np.pad(points[..., :3], ((0, 0), (0, 1)), constant_values=1)
                @ sensor2baselidar.T
            )[:, :3]
            point_with_intensity[:, :3] = points
            gt_points = gt_point_with_intensity[:, :3]
            gt_points = (
                np.pad(gt_points[..., :3], ((0, 0), (0, 1)), constant_values=1)
                @ sensor2baselidar.T
            )[:, :3]
            gt_point_with_intensity[:, :3] = gt_points

        np.savetxt(
            os.path.join(render_path, "{}.txt".format(render_timestamp)),
            point_with_intensity,
        )
        np.savetxt(
            os.path.join(gt_path, "{}.txt".format(render_timestamp)),
            gt_point_with_intensity,
        )

    # 计算总渲染时间
    total_render_time = time.time() - total_render_start

    # 计算帧率 (FPS = 总帧数 / 总时间)
    fps = total_frames / total_render_time if total_render_time > 0 else 0

    # 打印帧率信息
    print(f"\n渲染完成:")
    print(f"总渲染帧数: {total_frames}")
    print(f"总渲染时间: {total_render_time:.4f} 秒")
    print(f"平均帧率 (FPS): {fps:.4f}")

    # 计算单帧平均渲染时间
    if t_list:
        avg_frame_time = sum(t_list) / len(t_list)
        print(f"单帧平均渲染时间: {avg_frame_time:.4f} 秒")


def render_sets(
    gt_dynamic_model,
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    edit_obj_info,
    insert_objs=None,
    obj_types=None,
):
    model_id_list = [0]
    if gt_dynamic_model.get_dynamic_obj_id_list() is not None:
        model_id_list.extend(gt_dynamic_model.get_dynamic_obj_id_list())
    print("model_id_list ", model_id_list)
    if edit_obj_info is not None:
        for delete_id in edit_obj_info.delete_obj_ids:
            if delete_id in model_id_list:
                model_id_list.remove(delete_id)
        print("After deletion, model_id_list ", model_id_list)

    with torch.no_grad():
        static_views = None
        model_id_scene_info = {}

        for model_id in model_id_list:
            model_gaussians = GaussianModel(
                dataset.feat_dim,
                dataset.n_offsets,
                dataset.voxel_size,
                dataset.update_depth,
                dataset.update_init_factor,
                dataset.update_hierachy_factor,
                dataset.use_feat_bank,
                dataset.appearance_dim,
                dataset.ratio,
                dataset.add_opacity_dist,
                dataset.add_cov_dist,
                dataset.add_color_dist,
                dataset.color_channel,
            )
            model_scene = Scene(
                dataset,
                model_id,
                gt_dynamic_model,
                model_gaussians,
                load_iteration=iteration,
                shuffle=False,
            )

            if model_scene.init_status:

                time_poses = {}
                total_views = model_scene.getTotalCameras()
                print("total_views ", len(total_views))

                if model_id == 0:
                    static_views = total_views

                for view in total_views:
                    timestamp = view.image_name
                    camera_pose = np.eye(4)
                    camera_pose[:3, :3] = np.transpose(view.R)
                    camera_pose[:3, 3] = view.T
                    time_poses[timestamp] = TimePose(
                        camera_pose=camera_pose,
                        view=view,
                        valid_mask=view.img_mask,
                        gt_mask=view.original_image,
                    )
                transform_R = torch.eye(
                    3,
                    device=model_gaussians._anchor.device,
                    dtype=model_gaussians._anchor.dtype,
                )
                if edit_obj_info is not None:
                    for move_pair in edit_obj_info.obj_id_offset_pairs:
                        if move_pair[0] == model_id:
                            transform_T = torch.tensor(
                                move_pair[1],
                                device=model_gaussians._anchor.device,
                                dtype=model_gaussians._anchor.dtype,
                            )
                            new_anchor = (
                                model_gaussians._anchor @ transform_R.T
                            ) + transform_T.reshape(1, 3)
                            model_gaussians._anchor.data.copy_(new_anchor)
                model_id_scene_info[model_id] = GaussianView(
                    gaussians=model_gaussians, scene=model_scene, time_poses=time_poses
                )

            model_gaussians.eval()
        test_views = []
        test_timestamp = []
        for idx, scene_view in enumerate(static_views):
            render_timestamp = scene_view.image_name
            test_timestamp.append(int(render_timestamp))
            test_views.append(scene_view)
        valid_timestamp_model = {}
        for view in static_views:
            timestamp = view.image_name
            valid_timestamp_model[timestamp] = []
            for model_id in model_id_scene_info.keys():
                if timestamp in model_id_scene_info[model_id].time_poses:
                    valid_timestamp_model[timestamp].append(
                        model_id
                    )  # 记录每一帧timestep（0-50）下出现的obj的id

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        render_set(
            gt_dynamic_model,
            dataset,
            "test",
            iteration,
            valid_timestamp_model,
            model_id_scene_info,
            test_views,
            pipeline,
            background,
            insert_objs,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_frames", nargs="+", type=int, default=[])
    parser.add_argument("--edit_json", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    # 解析json文件
    if args.edit_json is not None:
        with open(args.edit_json, "r") as f:
            edit_args = json.load(f)
        for key, value in edit_args.items():
            print(f"Setting {key} to {value} from {args.edit_json}")
            setattr(args, key, value)
    if args.test_frames is None and args.novel_poses is None:
        print("Please set test_frames or novel_poses in edit_config.json")
        sys.exit(1)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    model_args = model.extract(args)

    if "chery" in args.model_path or "orinY" in args.model_path:
        from scene.chery_dataloader import Chery_Dataloader as GT_Dataloader
    else:
        print("ERROR: Unsupported data format.")
        logger.info("\nUnsupported data format.")
        sys.exit(1)

    train_frame_times = []
    pose_offsets = []
    if args.test_frames is not None:
        train_frame_times = args.test_frames
    else:
        for item in args.novel_poses:
            train_frame_times.append(item["frame_id"])
            pose_offsets.append(item["position_offset"])
    gt_dynamic_model = GT_Dataloader(
        model_args, train=False, train_frame_times=train_frame_times
    )
    edit_obj_info = EditObjInfo(
        delete_obj_ids=args.delete["obj_ids"] if hasattr(args, "delete") else [],
        obj_id_offset_pairs=(
            args.move["obj_id_offset_pairs"] if hasattr(args, "move") else []
        ),
        add_id_path_pairs=args.add["obj_id_path_pairs"] if hasattr(args, "add") else [],
    )

    model_args.block_id = 0
    objs = None
    render_sets(
        gt_dynamic_model,
        model_args,
        args.iteration,
        pipeline.extract(args),
        edit_obj_info,
        objs,
    )
