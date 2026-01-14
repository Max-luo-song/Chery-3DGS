#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的 UNet Refine 脚本
"""

import os
import sys
import torch
import numpy as np
import time
from tqdm import tqdm
from argparse import ArgumentParser
import torch.utils.data as data
from torch.cuda.amp import autocast, GradScaler

# 导入必要的模块
from arguments import ModelParams
from scene.unet import UNet
from skimage.metrics import structural_similarity
from utils.loss_utils import l1_loss
from utils.lidar_utils import PointsMeter
from utils.image_utils import psnr


def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if not logger.handlers:
        fileinfo = logging.FileHandler(os.path.join(path, "refine_outputs.log"))
        fileinfo.setLevel(logging.INFO)
        controlshow = logging.StreamHandler()
        controlshow.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        fileinfo.setFormatter(formatter)
        controlshow.setFormatter(formatter)
        logger.addHandler(fileinfo)
        logger.addHandler(controlshow)

    return logger


class RaydropDataset(data.Dataset):
    """自定义 Dataset 用于加载 raydrop 数据"""

    def __init__(self, train_dir, gt_dir):
        self.train_dir = train_dir
        self.gt_dir = gt_dir
        self.files = sorted([f for f in os.listdir(train_dir) if f.endswith(".pt")])

        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {train_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        # Load rendered data: [raydrop, intensity, depth]
        raydrop_input = torch.load(os.path.join(self.train_dir, filename))
        # 仅保留raydrop通道（输入只聚焦raydrop）
        raydrop_input = raydrop_input[[0]]  # 只取第0通道（raydrop）

        # Load GT raydrop (first channel)
        gt_loaded = torch.load(os.path.join(self.gt_dir, filename))

        if isinstance(gt_loaded, dict):
            gt_data = gt_loaded["gt_image"]
        else:
            gt_data = gt_loaded

        gt_raydrop = gt_data[[0]]
        return raydrop_input, gt_raydrop


# 掩码DiceLoss
def dice_loss_hard(pred_logits, target):
    pred_hard = torch.where(pred_logits > 0.5, 1.0, 0.0)
    smooth = 1e-6
    intersection = (pred_hard * target).sum()
    union = pred_hard.sum() + target.sum() + smooth
    return 1 - (2 * intersection + smooth) / union


def refine_with_dataloader(dataset, logger, use_amp=True):
    """
    使用 DataLoader + 混合精度训练 UNet
    """

    ray_drop_datasets_dir = os.path.join(dataset.model_path, "ray_drop_datasets")
    gt_dir = os.path.join(ray_drop_datasets_dir, "gt")
    train_dir = os.path.join(ray_drop_datasets_dir, "render_train")

    # 检查数据是否存在
    if not os.path.exists(train_dir) or not os.path.exists(gt_dir):
        logger.error(f"训练数据不存在！请先运行训练生成数据")
        logger.error(f"train_dir: {train_dir}")
        logger.error(f"gt_dir: {gt_dir}")
        return False

    # 创建 Dataset 和 DataLoader
    try:
        dataset_rd = RaydropDataset(train_dir, gt_dir)
        logger.info(f"成功加载 {len(dataset_rd)} 个训练样本")
    except ValueError as e:
        logger.error(str(e))
        return False

    refine_bs = 8
    dataloader = data.DataLoader(
        dataset_rd, batch_size=refine_bs, shuffle=True, num_workers=2, pin_memory=True
    )

    # UNet输入通道从3→1（仅输入raydrop）
    unet = UNet(in_channels=1, out_channels=1)
    unet.cuda()
    unet.train()

    # 混合精度训练
    scaler = GradScaler() if use_amp else None

    refine_epoch = dataset.unet_iterations
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=refine_epoch, eta_min=1e-6
    )

    bce_fn = torch.nn.BCEWithLogitsLoss()

    logger.info(
        f"开始 UNet 训练: batch_size={refine_bs}, epochs={refine_epoch}, AMP={use_amp}"
    )

    for epoch in range(refine_epoch):
        epoch_loss = 0.0

        for batch_idx, (input_batch, gt_batch) in enumerate(dataloader):
            input_batch = input_batch.cuda()
            gt_batch = gt_batch.cuda()

            optimizer.zero_grad()

            mask = torch.ones_like(input_batch)
            box_num_max = 4
            box_size_y_max = int(0.05 * input_batch.shape[2])
            box_size_x_max = int(0.05 * input_batch.shape[3])
            for j in range(np.random.randint(box_num_max)):
                box_size_y = np.random.randint(1, box_size_y_max)
                box_size_x = np.random.randint(1, box_size_x_max)
                yi = np.random.randint(input_batch.shape[2] - box_size_y)
                xi = np.random.randint(input_batch.shape[3] - box_size_x)
                mask[:, :, yi : yi + box_size_y, xi : xi + box_size_x] = 0.0

            # 前向传播（混合精度）
            if use_amp:
                with autocast():
                    raydrop_refine = unet(input_batch * mask)
                    # 组合Loss：优先保证raydrop掩码精度（Dice权重调高）
                    bce_loss = bce_fn(raydrop_refine.float(), gt_batch.float())
                    dice_loss_val = dice_loss_hard(
                        raydrop_refine.float(), gt_batch.float()
                    )
                    total_loss = bce_loss + 1.0 * dice_loss_val

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                raydrop_refine = unet(input_batch * mask)
                bce_loss = bce_fn(raydrop_refine, gt_batch)
                dice_loss_val = dice_loss_hard(raydrop_refine, gt_batch)
                total_loss = bce_loss + 1.2 * dice_loss_val

                total_loss.backward()
                optimizer.step()

            epoch_loss += total_loss.item()

            # 清理中间变量（避免显存泄漏）
            del input_batch, gt_batch, raydrop_refine, total_loss, mask

        scheduler.step()

        # 每 50 个 epoch 记录
        if epoch % 50 == 0:
            avg_loss = epoch_loss / len(dataloader)
            allocated = torch.cuda.memory_allocated() / 1024**3
            log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            unet.eval()
            iou_list = []
            with torch.no_grad():
                for test_input, test_gt in dataloader:
                    pred_logits = unet(test_input.cuda()).cpu()
                    pred_hard = torch.where(pred_logits > 0.5, 1.0, 0.0)
                    test_gt = test_gt.cpu()
                    intersection = (pred_hard * test_gt).sum()
                    union = (pred_hard + test_gt).clamp(0, 1).sum() + 1e-6
                    iou_list.append((intersection / union).item())
            avg_iou = np.mean(iou_list)
            unet.train()

            logger.info(
                f"[{log_time}] Epoch:{epoch:4d}, "
                f"lr:{optimizer.param_groups[0]['lr']:.6f}, "
                f"loss:{avg_loss:.6f}, "
                f"raydrop_IoU:{avg_iou:.4f}, "
                f"GPU:{allocated:.2f}GB"
            )

        torch.cuda.empty_cache()

    # 保存 UNet checkpoint
    ckpt_dir = os.path.join(dataset.model_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    unet_path = os.path.join(ckpt_dir, "unet_refine.pth")
    torch.save(unet.state_dict(), unet_path)
    logger.info(f"UNet 权重已保存到: {unet_path}")

    del unet
    torch.cuda.empty_cache()

    return True


def refine_test(dataset, logger):
    """
    测试 UNet refinement 效果
    """
    ckpt_path = os.path.join(dataset.model_path, "ckpt", "unet_refine.pth")
    if not os.path.exists(ckpt_path):
        logger.error(f"UNet checkpoint 不存在: {ckpt_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = UNet(in_channels=1, out_channels=1)
    state = torch.load(ckpt_path, map_location="cpu")
    unet.load_state_dict(state)
    unet.to(device)
    unet.eval()

    ray_drop_datasets_dir = os.path.join(dataset.model_path, "ray_drop_datasets")

    for mode in ["train", "test"]:
        test_dir = os.path.join(ray_drop_datasets_dir, f"render_{mode}")
        gt_dir = os.path.join(ray_drop_datasets_dir, "gt")

        if not os.path.exists(test_dir):
            logger.warning(f"{mode} 数据不存在，跳过: {test_dir}")
            continue

        logger.info(f"测试 {mode} 集 refinement 效果...")

        outdir = os.path.join(
            dataset.model_path, "eval", f"{mode}_refine_hard_cd_stable"
        )
        os.makedirs(outdir, exist_ok=True)
        test_dir = os.path.join(ray_drop_datasets_dir, f"render_{mode}")

        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".pt")])

        l1_test = 0.0
        psnr_test = 0.0
        in_mae = 0.0
        in_rmse = 0.0
        in_medae = 0.0
        in_ssim = 0.0
        cd_test = 0.0
        fscore_test = 0.0
        rmse = 0.0
        mae = 0.0
        medae = 0.0
        raydrop_iou = 0.0

        cd_original = 0.0

        for filename in tqdm(test_files, desc=f"Testing {mode}"):
            # 加载数据
            render_data = (
                torch.load(os.path.join(test_dir, filename)).unsqueeze(0).cuda()
            )
            render_raydrop = render_data[:, [0]]  # 只取raydrop通道给UNet

            # 提取原始raydrop的mask（用于对比CD）
            original_mask = torch.where(render_data[:, [0]] > 0.5, 1, 0)

            gt_loaded = torch.load(os.path.join(gt_dir, filename))

            beam_inclinations = None
            if isinstance(gt_loaded, dict):
                gt_data = gt_loaded["gt_image"].cuda()
                beam_inclinations = gt_loaded["beam_inclinations"].numpy()
            else:
                gt_data = gt_loaded.cuda()

            with torch.no_grad():
                # UNet refinement
                raydrop_refine = unet(render_raydrop)
                raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)
                gt_raydrop = gt_data[[0]]

            # 计算raydrop IoU
            pred_bin = raydrop_mask.cpu()
            gt_raydrop_bin = gt_data[[0]].cpu()
            intersection = (pred_bin * gt_raydrop_bin).sum()
            union = (pred_bin + gt_raydrop_bin).clamp(0, 1).sum() + 1e-6
            raydrop_iou += (intersection / union).item()

            # 提取通道
            gt_raydrop = gt_data[[0]]
            gt_objmask = 1.0
            gt_intensity = gt_data[[1]] * gt_raydrop
            gt_depth = gt_data[[2]] * gt_raydrop

            refined_intensity = render_data[0, [1]] * raydrop_mask[0]
            refined_depth = render_data[0, [2]] * raydrop_mask[0]

            # 计算原始raydrop的CD（对比优化效果）
            original_depth = render_data[0, [2]] * original_mask[0]
            if True:  # new trick: using depth_distortion_aware
                depth_distortion_aware = render_data[0, [3]]
                depth_distortion_aware = torch.where(depth_distortion_aware < 0.3, 1, 0)
                refined_depth = refined_depth * depth_distortion_aware
                original_depth = original_depth * depth_distortion_aware

            # 计算指标
            l1_test += l1_loss(refined_intensity, gt_intensity).item()
            psnr_test += psnr(refined_intensity, gt_intensity).mean().double().item()

            error_in_abs = torch.abs(refined_intensity - gt_intensity)
            in_mae += error_in_abs.mean().item()
            in_rmse += torch.sqrt((error_in_abs * error_in_abs).mean()).item()
            in_medae += error_in_abs.median().item()

            in_ssim += structural_similarity(
                refined_intensity[0].detach().cpu().numpy(),
                gt_intensity[0].detach().cpu().numpy(),
                data_range=1.0,
            )

            # Chamfer Distance and Fscore for depth
            if beam_inclinations is not None:
                if isinstance(beam_inclinations, torch.Tensor):
                    beam_inclinations = beam_inclinations.detach().cpu().numpy()
                else:
                    beam_inclinations = np.asarray(beam_inclinations)
                beam_inclinations = beam_inclinations.ravel()
                points_meter = PointsMeter(
                    scale=1,
                    intrinsics=None,
                    beam_inclinations=beam_inclinations,
                )
                # 优化后的CD
                points_meter.update(refined_depth, gt_depth, Filter=False)
                cd_fs = points_meter.measure()
                cd_test += cd_fs[0]
                fscore_test += cd_fs[1]

                # 原始raydrop的CD（对比）
                points_meter_ori = PointsMeter(
                    scale=1,
                    intrinsics=None,
                    beam_inclinations=beam_inclinations,
                )
                points_meter_ori.update(original_depth, gt_depth, Filter=False)
                cd_ori = points_meter_ori.measure()[0]
                cd_original += cd_ori

            error_depth_abs = torch.abs(refined_depth - gt_depth)
            mae += error_depth_abs.mean().item()
            rmse += torch.sqrt((error_depth_abs * error_depth_abs).mean()).item()
            medae += error_depth_abs.median().item()

        # 平均指标
        total = len(test_files)
        if total > 0:
            l1_test /= total
            psnr_test /= total
            in_mae /= total
            in_rmse /= total
            in_medae /= total
            in_ssim /= total
            cd_test /= total
            fscore_test /= total
            mae /= total
            rmse /= total
            medae /= total
            raydrop_iou /= total
            cd_original /= total

        logger.info(
            f"\n[{mode.upper()} REFINED] "
            f"raydrop_IoU:{raydrop_iou:.4f} | "
            f"CD(original):{cd_original:.2f}m | CD(refined):{cd_test:.2f}m | "  # 对比CD
            f"intensity: L1 {l1_test:.6f} PSNR {psnr_test:.6f} SSIM {in_ssim:.6f} MAE {in_mae:.6f} RMSE {in_rmse:.6f} MadAE {in_medae:.6f} // "
            f"depth: Fscore {fscore_test:.6f} MAE {mae:.6f} MedAE {medae:.6f} RMSE {rmse:.6f}"
        )

    del unet
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # 解析参数
    parser = ArgumentParser(description="UNet Refine script")
    lp = ModelParams(parser)
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--block_size", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="chery")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度训练")
    parser.add_argument("--skip_test", action="store_true", help="跳过测试阶段")
    args = parser.parse_args(sys.argv[1:])

    # 设置 GPU
    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"使用 GPU: {args.gpu}")

    model_args = lp.extract(args)
    logger = get_logger(model_args.model_path)

    logger.info("=" * 60)
    logger.info("开始 UNet Refine")
    logger.info("=" * 60)
    logger.info(f"数据路径: {model_args.source_path}")
    logger.info(f"模型路径: {model_args.model_path}")

    # 加载 dataloader（用于获取 block 信息）
    if args.dataset == "chery":
        from scene.chery_dataloader import Chery_Dataloader as GT_Dataloader
        from utils.data_partition_utils import dataPartitionChery

        block_info_with_extend, _ = dataPartitionChery(
            model_args, args.block_size, single_block_test=True
        )
    elif args.dataset == "zdrive":
        from scene.zdrive_dataloader import ZDrive_Dataloader as GT_Dataloader
        from utils.data_partition_utils import dataPartitionChery

        block_info_with_extend, _ = dataPartitionChery(
            model_args, args.block_size, single_block_test=True
        )
    else:
        logger.error("不支持的数据集类型")
        sys.exit(1)

    ray_drop_dir = os.path.join(model_args.model_path, "ray_drop_datasets")
    if not os.path.exists(ray_drop_dir):
        logger.error(f"训练数据不存在，跳过: {ray_drop_dir}")
        logger.error(f"请先运行 train.py 生成训练数据")
        sys.exit(1)
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # 训练 UNet
    success = refine_with_dataloader(model_args, logger, use_amp=args.use_amp)
    if not success:
        logger.error(f"refine 失败")
        sys.exit(1)

    # 测试 UNet
    if not args.skip_test:
        logger.info(f"\n测试 refinement 效果...")
        refine_test(model_args, logger)

    logger.info(f"refine 完成\n")

    # 清理显存
    torch.cuda.empty_cache()
