from omegaconf import OmegaConf
import numpy as np
import os
import time
import wandb
import random
import imageio
import logging
import argparse

import torch
from tools.eval import do_evaluation
from utils.misc import import_str
from utils.backup import backup_project
from utils.logging import MetricLogger, setup_logging
from models.video_utils import render_images, save_videos
from datasets.driving_dataset import DrivingDataset

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def clean_road_overhead_during_training(trainer, road_xyz_stats, height_limit=0.3, xy_margin=0.05):
    """
    在线清理路面上方的背景杂点。
    
    Args:
        bg_model: 背景的高斯模型对象
        road_xyz_stats: 一个字典，包含路面几何的统计信息 {'min_x', 'max_x', 'min_y', 'max_y', 'avg_z'}
        height_limit: 清理高度 (米)，路面以上多少米内是禁区
        xy_margin: 水平缩进 (米)，防止切掉路边的路缘石
    """
    # 获取背景点坐标 [N, 3]
    bg_model=trainer.models['Background']
    bg_xyz = bg_model._means
    
    # 1. 水平范围判断 (XY Plane)
    # 只处理位于路面垂直投影范围内的点
    mask_x = (bg_xyz[:, 0] > road_xyz_stats['min_x'] + xy_margin) & \
             (bg_xyz[:, 0] < road_xyz_stats['max_x'] - xy_margin)
    mask_y = (bg_xyz[:, 1] > road_xyz_stats['min_y'] + xy_margin) & \
             (bg_xyz[:, 1] < road_xyz_stats['max_y'] - xy_margin)
    in_road_footprint = mask_x & mask_y
    
    # 如果没有点在路面范围内，直接返回，节省计算
    if not in_road_footprint.any():
        return
        
    # 2. 垂直高度判断 (Z Axis)
    # 假设路面大致是一个平面，或者你已经将其转换到了 Z=0 附近
    # 逻辑：在路面以下 (z < road_z) 或者 路面以上 height_limit 内 (z < road_z + limit) 的背景点都要死
    # 这里的 min_z - 0.5 是为了把那种错误的地下点也顺手清了
    
    road_z = road_xyz_stats['avg_z'] # 或者使用更复杂的平面拟合 z = ax + by + c
    
    # 禁区：从地下 0.5m 到 路面上方 0.15m
    mask_z = (bg_xyz[:, 2] > road_z - 0.5) & (bg_xyz[:, 2] < road_z + height_limit)
    
    # 3. 最终死刑名单
    kill_mask = in_road_footprint & mask_z
    
    if kill_mask.sum() > 0:
        # 执行剔除
        bg_model.prune_points(kill_mask, optimizer=trainer.optimizer)
        # 这是一个可选的打印，调试时开启，平时关闭以免刷屏
        # print(f"Cleaned {kill_mask.sum()} interference points from background.")

def set_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup(args):
    # get config
    cfg = OmegaConf.load(args.config_file)
    
    # parse datasets
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")
        
    assert "dataset" in cfg or "data" in cfg, \
        "Please specify dataset in config or data in config"
        
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        )
        # merge data
        cfg = OmegaConf.merge(cfg, dataset_cfg)
    
    # merge cli
    cfg = OmegaConf.merge(cfg, args_from_cli)
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    
    # update config and create log dir
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for folder in ["images", "videos", "metrics", "configs_bk", "buffer_maps", "backup"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    
    # setup wandb
    if args.enable_wandb:
        # sometimes wandb fails to init in cloud machines, so we give it several (many) tries
        while (
            wandb.init(
                project=args.project,
                entity=args.entity,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="fork"),
            )
            is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    # setup random seeds
    set_seeds(cfg.seed)

    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
        
    # also save a backup copy
    # saved_cfg_path_bk = os.path.join(log_dir, "configs_bk", f"config_{current_time}.yaml")
    # with open(saved_cfg_path_bk, "w") as f:
    #     OmegaConf.save(config=cfg, f=f)
    # logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")
    
    # # Backup codes
    # backup_project(
    #     os.path.join(log_dir, 'backup'), "./", 
    #     ["configs", "datasets", "models", "utils", "tools"], 
    #     [".py", ".h", ".cpp", ".cuh", ".cu", ".sh", ".yaml"]
    # )
    return cfg

def main(args):
    cfg = setup(args)
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
        device=device
    )
    
    # NOTE: If resume, gaussians will be loaded from checkpoint
    #       If not, gaussians will be initialized from dataset
    if args.resume_from is not None:
        trainer.resume_from_checkpoint(
            ckpt_path=args.resume_from,
            load_only_model=True
        )
        logger.info(
            f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
        )
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
        logger.info(
            f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}"
        )
    
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # define render keys
    render_keys = [   ## 体现在output里面的images中，每隔几k就会输出
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "Dynamic_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        "depths",
        # "Background_depths",
        # "Dynamic_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        "gt_road_rgbs", ### TODO(gls): add something
        "road_rgbs" ### TODO(gls): add something
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    
    # setup optimizer  
    trainer.initialize_optimizer()
    
    # setup metric logger
    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    all_iters = np.arange(trainer.step, trainer.num_iters + 1)
    
    # DEBUG USE
    # do_evaluation(
    #     step=0,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    # )

    # ------------------  Preparation for Geometric Pruning  ---------------------
    # 假设你的 Road 模型名字叫 'Road'，请根据实际情况修改 key
    if 'RoadNodes' in trainer.gaussian_classes:
        road_model = trainer.models['RoadNodes']
        road_xyz = road_model._means.detach()
        
        # 预计算路面包围盒信息
        road_stats = {
            'min_x': road_xyz[:, 0].min().item(),
            'max_x': road_xyz[:, 0].max().item(),
            'min_y': road_xyz[:, 1].min().item(),
            'max_y': road_xyz[:, 1].max().item(),
            # 假设路面大致平坦，取平均高度；如果有坡度，后续逻辑需要改为 KNN
            'avg_z': road_xyz[:, 2].mean().item() 
        }
        print(f"Road Geometry Loaded: Z~={road_stats['avg_z']:.2f}, X[{road_stats['min_x']:.1f}, {road_stats['max_x']:.1f}]")
    else:
        road_stats = None
        print("Warning: Road model not found, skipping geometric pruning setup.")

    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        #----------------------------------------------------------------------------
        #----------------------------     Validate     ------------------------------
        if step % cfg.logging.vis_freq == 0 and cfg.logging.vis_freq > 0:
            logger.info("Visualizing...")
            vis_timestep = np.linspace(
                0,
                dataset.num_img_timesteps,
                trainer.num_iters // cfg.logging.vis_freq + 1,
                endpoint=False,
                dtype=int,
            )[step // cfg.logging.vis_freq]

            with torch.no_grad():
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                    compute_metrics=True,
                    compute_error_map=cfg.render.vis_error,
                    vis_indices=[
                        vis_timestep * dataset.pixel_source.num_cams + i
                        for i in range(dataset.pixel_source.num_cams)
                    ],
                )

            if args.enable_wandb:
                wandb.log(
                    {
                        "image_metrics/psnr": render_results["psnr"],
                        "image_metrics/ssim": render_results["ssim"],
                        "image_metrics/occupied_psnr": render_results["occupied_psnr"],
                        "image_metrics/occupied_ssim": render_results["occupied_ssim"],
                    }
                )

            vis_frame_dict = save_videos(
                render_results,
                save_pth=os.path.join(
                    cfg.log_dir, "images", f"step_{step}.png"
                ),  # don't save the video
                layout=dataset.layout,
                num_timestamps=1,
                keys=render_keys,
                save_seperate_video=cfg.logging.save_seperate_video,
                num_cams=dataset.pixel_source.num_cams,
                fps=cfg.render.fps,
                verbose=False,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"image_rendering/" + k: wandb.Image(v)})
            del render_results
            torch.cuda.empty_cache()
                
        #----------------------------------------------------------------------------
        #----------------------------  training step  -------------------------------
        # prepare for training
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad() # zero grad
        
        # get data
        train_step_camera_downscale = trainer._get_downscale_factor()
        image_infos, cam_infos = dataset.train_image_set.next(train_step_camera_downscale)
        for k, v in image_infos.items():
            if isinstance(v, torch.Tensor):
                image_infos[k] = v.cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.cuda(non_blocking=True)
        
        # forward & backward
        # drivestudio/models/trainers/scene_graph.py
        outputs, gs = trainer(image_infos, cam_infos)
        trainer.update_visibility_filter()

        loss_dict = trainer.compute_losses(
            outputs=outputs,
            image_infos=image_infos,
            cam_infos=cam_infos,
            gs=gs 
        )
        # check nan or inf
        for k, v in loss_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in loss {k} at step {step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in loss {k} at step {step}")
        trainer.backward(loss_dict)
        
        # after training step
        trainer.postprocess_per_train_step(step=step)
        
        # ----------------------------------------------------------------------------
        # -----------------------  Geometric Pruning Step  ---------------------------
        # 建议在一定步数后开始清理（比如 500 步后），并每隔几百步执行一次
        # 这里的 'Background' 是你背景模型的 key，请确保和 trainer.models 里的 key 一致
        if 'RoadNodes' in trainer.gaussian_classes and 'Background' in trainer.gaussian_classes:
            # 只有在做了致密化之后清理才有意义，或者每隔 200~500 step
            if step > 500 and step % 500 == 0:
            # if step > 0:
                print("发生过滤～")
                clean_road_overhead_during_training(
                    trainer = trainer,
                    road_xyz_stats=road_stats,
                    height_limit=0.3,  # 清理路面上方 15cm 内的杂点
                    xy_margin=0.05      # 稍微向内收缩，防止切到路沿
                )
                
                # 可选：清理完后清理显存，防止碎片化
                torch.cuda.empty_cache()
        #----------------------------------------------------------------------------
        #-------------------------------  logging  ----------------------------------
        with torch.no_grad():
            # cal stats
            metric_dict = trainer.compute_metrics(
                outputs=outputs,
                image_infos=image_infos,
            )
        metric_logger.update(**{"train_metrics/"+k: v.item() for k, v in metric_dict.items()})
        metric_logger.update(**{"train_stats/gaussian_num_" + k: v for k, v in trainer.get_gaussian_count().items()})
        metric_logger.update(**{"losses/"+k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{"train_stats/lr_" + group['name']: group['lr'] for group in trainer.optimizer.param_groups})
        if args.enable_wandb:
            wandb.log({k: v.avg for k, v in metric_logger.meters.items()})

        #----------------------------------------------------------------------------
        #----------------------------     Saving     --------------------------------
        do_save = step > 0 and (
            (step % cfg.logging.saveckpt_freq == 0) or (step == trainer.num_iters)
        ) and (args.resume_from is None)
        if do_save:  
            trainer.save_checkpoint(
                log_dir=cfg.log_dir,
                save_only_model=True,
                is_final=step == trainer.num_iters,
            )
        
        #----------------------------------------------------------------------------
        #------------------------    Cache Image Error    ---------------------------
        if (
            step > 0 and trainer.optim_general.cache_buffer_freq > 0
            and step % trainer.optim_general.cache_buffer_freq == 0
        ):
            logger.info("Caching image error...")
            trainer.set_eval()
            with torch.no_grad():
                dataset.pixel_source.update_downscale_factor(
                    1 / dataset.pixel_source.buffer_downscale
                )
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                )
                dataset.pixel_source.reset_downscale_factor()
                dataset.pixel_source.update_image_error_maps(render_results)

                # save error maps
                merged_error_video = dataset.pixel_source.get_image_error_video(
                    dataset.layout
                )
                imageio.mimsave(
                    os.path.join(
                        cfg.log_dir, "buffer_maps", f"buffer_maps_{step}.mp4"
                    ),
                    merged_error_video,
                    fps=cfg.render.fps,
                )
            logger.info("Done caching rgb error maps")
            
    
    logger.info("Training done!")

    do_evaluation(
        step=step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
    )
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)
    
    return step

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument("--output_root", default="./work_dirs/", help="path to save checkpoints and logs", type=str)
    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    
    # wandb logging part
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--entity", default="ziyc", type=str, help="wandb entity name")
    parser.add_argument("--project", default="drivestudio", type=str, help="wandb project name, also used to enhance log_dir")
    parser.add_argument("--run_name", default="omnire", type=str, help="wandb run name, also used to enhance log_dir")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
    
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    final_step = main(args)
