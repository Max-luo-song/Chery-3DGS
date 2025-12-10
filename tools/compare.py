# 对比两次训练输出的评估结果
# 找出指标变好的情况，并展示变化最大的帧
import os
import json
import argparse
import glob
import numpy as np
import wandb
import imageio
import re
import cv2
import csv
from omegaconf import OmegaConf
from typing import Dict, List, Tuple, Optional, Set
import logging

logger = logging.getLogger()

# 全局 LPIPS 模型实例（只加载一次）
_lpips_model = None

def get_lpips_model():
    """获取全局 LPIPS 模型实例，只加载一次"""
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips
            _lpips_model = lpips.LPIPS(net='alex')
            logger.info("LPIPS model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load LPIPS model: {e}")
            _lpips_model = None
    return _lpips_model

# 需要对比的17个指标
METRICS_TO_COMPARE = [
    "image_metrics/full/psnr",
    "image_metrics/full/ssim",
    "image_metrics/full/lpips",
    "image_metrics/full/psnr_no_ego",
    "image_metrics/full/ssim_no_ego",
    "image_metrics/full/lpips_no_ego",
    "image_metrics/full/psnr_with_ego",
    "image_metrics/full/ssim_with_ego",
    "image_metrics/full/lpips_with_ego",
    "image_metrics/full/occupied_psnr",
    "image_metrics/full/occupied_ssim",
    "image_metrics/full/masked_psnr",
    "image_metrics/full/masked_ssim",
    "image_metrics/full/human_psnr",
    "image_metrics/full/human_ssim",
    "image_metrics/full/vehicle_psnr",
    "image_metrics/full/vehicle_ssim",
]


def load_metrics_file(metrics_file: str) -> Dict:
    # 加载评估指标文件
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def find_latest_metrics_file(log_dir: str, prefix: str = "images_full") -> Optional[str]:
    # 查找最新的评估指标文件
    metrics_dir = os.path.join(log_dir, "metrics")
    if not os.path.exists(metrics_dir):
        # 尝试 metrics_eval
        metrics_dir = os.path.join(log_dir, "metrics_eval")
        if not os.path.exists(metrics_dir):
            return None
    
    pattern = os.path.join(metrics_dir, f"{prefix}_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    
    # 返回最新的文件（按修改时间）
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def get_camera_ids_from_config(log_dir: str) -> Optional[List[int]]:
    """
    从 config.yaml 文件中读取相机配置
    返回相机ID列表，如果无法读取则返回None
    """
    config_path = os.path.join(log_dir, "config.yaml")
    if not os.path.exists(config_path):
        return None
    
    try:
        cfg = OmegaConf.load(config_path)
        # 尝试从 data.pixel_source.cameras 获取
        if hasattr(cfg, 'data') and hasattr(cfg.data, 'pixel_source') and hasattr(cfg.data.pixel_source, 'cameras'):
            cameras = cfg.data.pixel_source.cameras
            
            # 处理 OmegaConf 的 ListConfig 类型
            if cameras is not None:
                try:
                    # 尝试直接转换为列表（支持 list, tuple, 和 OmegaConf ListConfig）
                    cam_list = list(cameras)
                    # 确保所有元素都是整数
                    return [int(cam) for cam in cam_list if cam is not None]
                except (TypeError, ValueError) as e:
                    # 如果转换失败，尝试作为字符串处理
                    if isinstance(cameras, str):
                        return [int(x.strip()) for x in cameras.split(',') if x.strip()]
                    logger.debug(f"无法解析相机配置: {e}")
    except Exception as e:
        logger.warning(f"Failed to load camera config from {config_path}: {e}")
    
    return None


def get_camera_ids_from_run_name(run_name: str) -> Optional[List[int]]:
    """
    从 run_name 中解析相机ID
    例如: "202511240615_lidar+cam0_1_2_3_4_5_7_8_10" -> [0, 1, 2, 3, 4, 5, 7, 8, 10]
    """
    # 匹配 cam0_1_2_3 或 cam0,1,2,3 或 cam0_1_2_3_4_5_7_8_10 等格式
    pattern = r'cam(\d+(?:[_,]\d+)*)'
    match = re.search(pattern, run_name)
    if match:
        cam_str = match.group(1)
        # 处理下划线和逗号分隔
        cam_ids = []
        for part in re.split(r'[_,]', cam_str):
            try:
                cam_ids.append(int(part))
            except ValueError:
                continue
        return cam_ids if cam_ids else None
    return None


def get_camera_ids(log_dir: str) -> Optional[List[int]]:
    """
    获取训练使用的相机ID列表
    优先从 config.yaml 读取，如果失败则从 run_name 解析
    """
    # 方法1: 从 config.yaml 读取
    cam_ids = get_camera_ids_from_config(log_dir)
    if cam_ids:
        return cam_ids
    
    # 方法2: 从 run_name 解析
    run_name = os.path.basename(log_dir)
    cam_ids = get_camera_ids_from_run_name(run_name)
    if cam_ids:
        return cam_ids
    
    logger.warning(f"Could not determine camera IDs from {log_dir}")
    return None


def get_common_camera_ids(cam_ids1: Optional[List[int]], cam_ids2: Optional[List[int]]) -> Set[int]:
    """
    获取两次训练共同的相机ID集合
    """
    if cam_ids1 is None or cam_ids2 is None:
        return set()
    
    set1 = set(cam_ids1)
    set2 = set(cam_ids2)
    common = set1 & set2
    
    return common


def compare_metrics(metrics1: Dict, metrics2: Dict) -> Dict[str, Dict]:
    # 对比两次评估的指标
    # 返回: {metric_name: {"old": value1, "new": value2, "diff": diff, "improved": bool}}
    comparison = {}
    
    for metric in METRICS_TO_COMPARE:
        if metric not in metrics1 or metric not in metrics2:
            logger.warning(f"Metric {metric} not found in one or both results")
            continue
        
        old_value = metrics1[metric]
        new_value = metrics2[metric]
        
        # 判断指标是否变好
        # 对于 PSNR，越大越好；对于 SSIM，越大越好；对于 LPIPS，越小越好
        if "lpips" in metric.lower():
            improved = new_value < old_value  # LPIPS 越小越好，数值降低是提升
            diff = old_value - new_value  # 正数表示改进（LPIPS降低）
            # 对于LPIPS，improvement_pct应该基于降低的绝对值，但保持正数表示改进
            improvement_pct = (diff / old_value * 100) if old_value != 0 else 0
        else:
            improved = new_value > old_value  # PSNR/SSIM 越大越好
            diff = new_value - old_value  # 正数表示改进
            improvement_pct = (diff / old_value * 100) if old_value != 0 else 0
        
        comparison[metric] = {
            "old": old_value,
            "new": new_value,
            "diff": diff,
            "improved": improved,
            "improvement_pct": improvement_pct
        }
    
    return comparison


def extract_frame_cam_from_filename(filename: str) -> Optional[Tuple[int, int]]:
    """
    从文件名中提取帧ID和相机ID
    支持格式:
    1. {frame:03d}_{cam_id:03d}.png (例如: 000_003.png)
    2. frame_{timestamp:06d}_cam{cam_id}_{key}.png (例如: frame_000000_cam3_rgbs.png)
    
    返回: (frame_id, cam_id) 或 None
    """
    # 格式1: 000_003.png
    match1 = re.match(r'(\d+)_(\d+)\.png', filename)
    if match1:
        try:
            frame_id = int(match1.group(1))
            cam_id = int(match1.group(2))
            return (frame_id, cam_id)
        except:
            pass
    
    # 格式2: frame_000000_cam3_rgbs.png
    match2 = re.search(r'frame_(\d+).*cam(\d+)', filename)
    if match2:
        try:
            frame_id = int(match2.group(1))
            cam_id = int(match2.group(2))
            return (frame_id, cam_id)
        except:
            pass
    
    return None


def compute_frame_metrics_score(
    img_old_path: str,
    img_new_path: str,
    gt_img_path: Optional[str] = None,
    metrics_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    计算单帧的综合指标分数，基于17个指标
    必须有GT图像才能计算，否则返回0.0
    """
    try:
        import torch
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        # 必须有GT图像
        if not gt_img_path or not os.path.exists(gt_img_path):
            logger.debug(f"No GT image provided for frame comparison, skipping")
            return 0.0
        
        img_old = imageio.imread(img_old_path)
        img_new = imageio.imread(img_new_path)
        gt_img = imageio.imread(gt_img_path)
        
        if img_old is None or img_new is None or gt_img is None:
            return 0.0
        
        # 确保图像尺寸一致
        if img_old.shape != img_new.shape or img_old.shape[:2] != gt_img.shape[:2]:
            h = min(img_old.shape[0], img_new.shape[0], gt_img.shape[0])
            w = min(img_old.shape[1], img_new.shape[1], gt_img.shape[1])
            img_old = cv2.resize(img_old, (w, h))
            img_new = cv2.resize(img_new, (w, h))
            gt_img = cv2.resize(gt_img, (w, h))
        
        # 归一化到[0,1]
        img_old_norm = img_old.astype(float) / 255.0
        img_new_norm = img_new.astype(float) / 255.0
        gt_norm = gt_img.astype(float) / 255.0
        
        scores = {}
        
        # PSNR
        psnr_old = psnr(gt_norm, img_old_norm, data_range=1.0)
        psnr_new = psnr(gt_norm, img_new_norm, data_range=1.0)
        scores['psnr'] = psnr_new - psnr_old
        
        # SSIM
        ssim_old = ssim(gt_norm, img_old_norm, data_range=1.0, channel_axis=2 if len(gt_norm.shape) == 3 else None)
        ssim_new = ssim(gt_norm, img_new_norm, data_range=1.0, channel_axis=2 if len(gt_norm.shape) == 3 else None)
        scores['ssim'] = ssim_new - ssim_old
        
        # LPIPS (使用全局模型实例，避免重复加载)
        try:
            loss_fn = get_lpips_model()
            if loss_fn is not None:
                with torch.no_grad():
                    gt_tensor = torch.from_numpy(gt_norm).permute(2, 0, 1).unsqueeze(0).float()
                    old_tensor = torch.from_numpy(img_old_norm).permute(2, 0, 1).unsqueeze(0).float()
                    new_tensor = torch.from_numpy(img_new_norm).permute(2, 0, 1).unsqueeze(0).float()
                    
                    lpips_old = loss_fn(gt_tensor, old_tensor).item()
                    lpips_new = loss_fn(gt_tensor, new_tensor).item()
                    scores['lpips'] = lpips_old - lpips_new  # LPIPS越小越好，所以是old-new
            else:
                scores['lpips'] = 0.0
        except Exception as e:
            logger.debug(f"Error computing LPIPS: {e}")
            scores['lpips'] = 0.0
        
        # 对于区域指标，由于没有区域信息，使用全图指标的近似值
        for suffix in ['_no_ego', '_with_ego', '_occupied', '_masked', '_human', '_vehicle']:
            scores[f'psnr{suffix}'] = scores['psnr'] * 0.8
            scores[f'ssim{suffix}'] = scores['ssim'] * 0.8
            if suffix not in ['_no_ego', '_with_ego']:
                scores[f'lpips{suffix}'] = scores['lpips'] * 0.8
        
        # 使用权重计算综合分数
        if metrics_weights is None:
            # 默认权重（基于17个指标的重要性）
            metrics_weights = {
                'psnr': 0.15, 'ssim': 0.15, 'lpips': 0.10,
                'psnr_no_ego': 0.08, 'ssim_no_ego': 0.08, 'lpips_no_ego': 0.05,
                'psnr_with_ego': 0.05, 'ssim_with_ego': 0.05, 'lpips_with_ego': 0.03,
                'occupied_psnr': 0.06, 'occupied_ssim': 0.06,
                'masked_psnr': 0.04, 'masked_ssim': 0.04,
                'human_psnr': 0.02, 'human_ssim': 0.02,
                'vehicle_psnr': 0.02, 'vehicle_ssim': 0.02,
            }
        
        # 归一化并加权
        total_score = 0.0
        total_weight = 0.0
        for metric_name, weight in metrics_weights.items():
            if metric_name in scores:
                # 归一化到[-1, 1]范围
                normalized_score = np.tanh(scores[metric_name])
                total_score += normalized_score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"Error computing frame metrics: {e}")
        return 0.0


def compute_overall_score(metrics: Dict, metrics_weights: Optional[Dict[str, float]] = None) -> float:
    """
    根据权重计算评估结果的综合分数
    
    Args:
        metrics: 指标字典，格式为 {"image_metrics/full/psnr": 25.5, ...}
        metrics_weights: 指标权重字典，如果为None则使用默认权重
        
    Returns:
        综合分数（float）
    """
    if metrics_weights is None:
        # 默认权重（基于17个指标的重要性）
        metrics_weights = {
            'psnr': 0.15, 'ssim': 0.15, 'lpips': 0.10,
            'psnr_no_ego': 0.08, 'ssim_no_ego': 0.08, 'lpips_no_ego': 0.05,
            'psnr_with_ego': 0.05, 'ssim_with_ego': 0.05, 'lpips_with_ego': 0.03,
            'occupied_psnr': 0.06, 'occupied_ssim': 0.06,
            'masked_psnr': 0.04, 'masked_ssim': 0.04,
            'human_psnr': 0.02, 'human_ssim': 0.02,
            'vehicle_psnr': 0.02, 'vehicle_ssim': 0.02,
        }
    
    total_score = 0.0
    total_weight = 0.0
    
    for metric in METRICS_TO_COMPARE:
        if metric not in metrics:
            continue
        
        metric_key = metric.split('/')[-1]  # 提取指标名称，如 "psnr", "ssim", "lpips"
        
        # 获取权重
        weight = metrics_weights.get(metric_key, 0.0)
        if weight == 0.0:
            continue
        
        value = metrics[metric]
        
        # 对于 LPIPS，需要取负值（因为越小越好）
        if "lpips" in metric.lower():
            value = -value  # 转换为越大越好的形式
        
        # 加权求和
        total_score += value * weight
        total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0.0


def find_best_improved_frames_per_camera(
    log_dir1: str,
    log_dir2: str,
    common_cam_ids: Optional[Set[int]] = None,
    metrics1: Optional[Dict] = None,
    metrics2: Optional[Dict] = None,
    reverse: bool = False
) -> Dict[int, Dict]:
    """
    对每个相机ID分别找出指标提升最大的帧（或下降最大的帧，如果reverse=True）
    返回: {cam_id: {"frame_idx": int, "cam_id": int, "image_path_old": str, "image_path_new": str, "improvement_score": float}}
    
    Args:
        reverse: 如果为True，找出新结果比旧结果差的帧（即旧结果相对新结果提升最大的帧）
    """
    def find_images_dir(base_dir: str) -> Optional[str]:
        """查找 full_set_*_rgbs 目录（videos_eval 优先，其次 videos）"""
        patterns = [
            os.path.join(base_dir, "videos_eval", "full_set_*_rgbs"),
            os.path.join(base_dir, "videos", "full_set_*_rgbs"),
        ]
        for pattern in patterns:
            dirs = glob.glob(pattern)
            if dirs:
                def extract_number(path):
                    match = re.search(r'full_set_(\d+)_rgbs', path)
                    return int(match.group(1)) if match else 0
                dirs.sort(key=extract_number, reverse=True)
                return dirs[0]
        return None

    def group_images_by_cam_frame(files, common_ids):
        grouped = {}
        for f in files:
            result = extract_frame_cam_from_filename(os.path.basename(f))
            if result:
                frame_id, cam_id = result
                if common_ids is None or cam_id in common_ids:
                    grouped[(cam_id, frame_id)] = f
        return grouped

    def find_gt_dir(base_dir: str) -> Optional[str]:
        """查找 full_set_*_gt_rgbs 目录（videos_eval 优先，其次 videos）"""
        patterns = [
            os.path.join(base_dir, "videos_eval", "full_set_*_gt_rgbs"),
            os.path.join(base_dir, "videos", "full_set_*_gt_rgbs"),
        ]
        for pattern in patterns:
            dirs = glob.glob(pattern)
            if dirs:
                def extract_number(path):
                    match = re.search(r'full_set_(\d+)_gt_rgbs', path)
                    return int(match.group(1)) if match else 0
                dirs.sort(key=extract_number, reverse=True)
                return dirs[0]
        return None

    def load_gt_grouped(gt_dir: Optional[str], common_ids) -> Dict[Tuple[int, int], str]:
        grouped = {}
        if not gt_dir:
            return grouped
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
        for f in gt_files:
            result = extract_frame_cam_from_filename(os.path.basename(f))
            if result:
                frame_id, cam_id = result
                if common_ids is None or cam_id in common_ids:
                    grouped[(cam_id, frame_id)] = f
        return grouped

    def build_metrics_weights(metrics1, metrics2):
        if not (metrics1 and metrics2):
            return None
        total_improvement = {}
        for metric in METRICS_TO_COMPARE:
            if metric in metrics1 and metric in metrics2:
                old_val = metrics1[metric]
                new_val = metrics2[metric]
                if "lpips" in metric.lower():
                    improvement = old_val - new_val  # LPIPS 越小越好
                else:
                    improvement = new_val - old_val  # PSNR/SSIM 越大越好
                total_improvement[metric] = improvement
        if not total_improvement:
            return None
        max_improvement = max(abs(v) for v in total_improvement.values())
        if max_improvement <= 0:
            return None
        weights = {}
        for metric in METRICS_TO_COMPARE:
            metric_key = metric.split('/')[-1]
            if metric in total_improvement:
                weight = abs(total_improvement[metric]) / max_improvement
                weights[metric_key] = weight
            else:
                weights[metric_key] = 0.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        return weights

    def compute_improvement_scores(common_keys, grouped1, grouped2, gt_grouped1, gt_grouped2, reverse, metrics_weights):
        frames_by_cam = {}
        total_frames = len(common_keys)
        processed_frames = 0
        logger.info(f"Processing {total_frames} frames across {len(set(cam_id for cam_id, _ in common_keys))} cameras...")
        for cam_id, frame_id in common_keys:
            frames_by_cam.setdefault(cam_id, [])
            img1_path = grouped1[(cam_id, frame_id)]
            img2_path = grouped2[(cam_id, frame_id)]
            gt_path = gt_grouped1.get((cam_id, frame_id)) or gt_grouped2.get((cam_id, frame_id))

            if reverse:
                # log_dir1 相对于 log_dir2 的改进
                improvement_score = compute_frame_metrics_score(img2_path, img1_path, gt_path, metrics_weights)
            else:
                # log_dir2 相对于 log_dir1 的改进
                improvement_score = compute_frame_metrics_score(img1_path, img2_path, gt_path, metrics_weights)

            frames_by_cam[cam_id].append((frame_id, improvement_score))
            processed_frames += 1
            if processed_frames % 50 == 0 or processed_frames == total_frames:
                logger.info(f"Processed {processed_frames}/{total_frames} frames ({100*processed_frames/total_frames:.1f}%)")
        return frames_by_cam

    def select_best_frames(frames_by_cam, grouped1, grouped2):
        best_frames = {}
        for cam_id, frame_scores in frames_by_cam.items():
            if not frame_scores:
                continue
            frame_scores.sort(key=lambda x: x[1], reverse=True)
            best_frame_id, best_improvement = frame_scores[0]
            key = (cam_id, best_frame_id)
            if key in grouped1 and key in grouped2:
                best_frames[cam_id] = {
                    "frame_idx": best_frame_id,
                    "cam_id": cam_id,
                    "image_path_old": grouped1[key],
                    "image_path_new": grouped2[key],
                    "improvement_score": best_improvement
                }
                logger.info(f"Cam {cam_id}: Best improved frame {best_frame_id} with improvement score {best_improvement:.4f}")
        return best_frames

    # 主流程
    images_dir1 = find_images_dir(log_dir1)
    images_dir2 = find_images_dir(log_dir2)
    if not images_dir1 or not images_dir2:
        logger.warning(f"Cannot find image directories for comparison")
        logger.warning(f"  Dir1: {images_dir1}, Dir2: {images_dir2}")
        return {}

    logger.info(f"Found image directories:")
    logger.info(f"  Run 1: {images_dir1}")
    logger.info(f"  Run 2: {images_dir2}")

    image_files1 = sorted(glob.glob(os.path.join(images_dir1, "*.png")))
    image_files2 = sorted(glob.glob(os.path.join(images_dir2, "*.png")))
    if not image_files1 or not image_files2:
        logger.warning(f"No image files found in directories")
        return {}

    grouped1 = group_images_by_cam_frame(image_files1, common_cam_ids)
    grouped2 = group_images_by_cam_frame(image_files2, common_cam_ids)
    common_keys = set(grouped1.keys()) & set(grouped2.keys())

    gt_dir1 = find_gt_dir(log_dir1)
    gt_dir2 = find_gt_dir(log_dir2)
    gt_grouped1 = load_gt_grouped(gt_dir1, common_cam_ids)
    gt_grouped2 = load_gt_grouped(gt_dir2, common_cam_ids)

    metrics_weights = build_metrics_weights(metrics1, metrics2)

    logger.info("Preloading LPIPS model...")
    get_lpips_model()

    frames_by_cam = compute_improvement_scores(
        common_keys, grouped1, grouped2, gt_grouped1, gt_grouped2, reverse, metrics_weights
    )

    best_frames = select_best_frames(frames_by_cam, grouped1, grouped2)
    return best_frames


def main(args):
    # 初始化 wandb
    if args.enable_wandb:
        wandb.init(
            project=args.wandb_project or "scene_recon_compare",
            name=args.wandb_run_name or f"compare_{os.path.basename(args.log_dir1)}_{os.path.basename(args.log_dir2)}",
            entity=args.wandb_entity or "1zzhaozz-nanjing-university",
        )
    
    # 获取两次训练的相机配置
    logger.info("Detecting camera configurations...")
    cam_ids1 = get_camera_ids(args.log_dir1)
    cam_ids2 = get_camera_ids(args.log_dir2)
    
    if cam_ids1:
        logger.info(f"Run 1 camera IDs: {sorted(cam_ids1)}")
    else:
        logger.warning(f"Could not detect camera IDs for {args.log_dir1}")
    
    if cam_ids2:
        logger.info(f"Run 2 camera IDs: {sorted(cam_ids2)}")
    else:
        logger.warning(f"Could not detect camera IDs for {args.log_dir2}")
    
    # 获取共同的相机ID
    common_cam_ids = get_common_camera_ids(cam_ids1, cam_ids2)
    
    if cam_ids1 and cam_ids2:
        if len(common_cam_ids) == 0:
            logger.warning("No common camera IDs found! Comparison may not be meaningful.")
        elif len(common_cam_ids) < len(set(cam_ids1)) or len(common_cam_ids) < len(set(cam_ids2)):
            logger.info(f"Common camera IDs: {sorted(common_cam_ids)}")
            logger.info(f"Run 1 only cameras: {sorted(set(cam_ids1) - common_cam_ids)}")
            logger.info(f"Run 2 only cameras: {sorted(set(cam_ids2) - common_cam_ids)}")
            logger.info("Will only compare images from common cameras.")
        else:
            logger.info(f"Both runs use the same camera configuration: {sorted(common_cam_ids)}")
    
    # 加载两次评估的指标
    logger.info(f"Loading metrics from {args.log_dir1}")
    metrics_file1 = find_latest_metrics_file(args.log_dir1, "images_full")
    if not metrics_file1:
        raise FileNotFoundError(f"Cannot find metrics file in {args.log_dir1}")
    
    logger.info(f"Loading metrics from {args.log_dir2}")
    metrics_file2 = find_latest_metrics_file(args.log_dir2, "images_full")
    if not metrics_file2:
        raise FileNotFoundError(f"Cannot find metrics file in {args.log_dir2}")
    
    metrics1 = load_metrics_file(metrics_file1)
    metrics2 = load_metrics_file(metrics_file2)
    
    logger.info(f"Metrics file 1: {metrics_file1}")
    logger.info(f"Metrics file 2: {metrics_file2}")
    
    # 只处理全图指标（image_metrics/full/）
    full_metrics = [m for m in METRICS_TO_COMPARE if m.startswith("image_metrics/full/")]
    
    # 对比指标
    comparison = compare_metrics(metrics1, metrics2)
    
    # 只保留全图指标的对比结果
    full_comparison = {k: v for k, v in comparison.items() if k in full_metrics}
    
    # 找出改进的指标
    improved_metrics = {k: v for k, v in full_comparison.items() if v["improved"]}
    
    logger.info(f"\n{'='*80}")
    logger.info(f"指标对比结果（仅全图指标）")
    logger.info(f"{'='*80}")
    logger.info(f"总共对比 {len(full_comparison)} 个全图指标")
    logger.info(f"改进的指标: {len(improved_metrics)} 个")
    logger.info(f"退化的指标: {len(full_comparison) - len(improved_metrics)} 个")
    logger.info(f"\n改进的指标详情:")
    
    # 按改进幅度排序
    sorted_improved = sorted(
        improved_metrics.items(),
        key=lambda x: abs(x[1]["diff"]),
        reverse=True
    )
    
    wandb_data = {}
    wandb_images = {}
    
    # 提取文件夹名称用于显示
    dir1_name = os.path.basename(args.log_dir1.rstrip('/'))
    dir2_name = os.path.basename(args.log_dir2.rstrip('/'))
    
    # 计算两次评估结果的综合分数
    logger.info(f"\n计算两次评估结果的综合分数...")
    score1 = compute_overall_score(metrics1)
    score2 = compute_overall_score(metrics2)
    
    logger.info(f"{dir1_name} 评估综合分数: {score1:.6f}")
    logger.info(f"{dir2_name} 评估综合分数: {score2:.6f}")
    
    # 记录综合分数到 wandb
    if args.enable_wandb:
        wandb_data[f"overall_score/{dir1_name}"] = score1
        wandb_data[f"overall_score/{dir2_name}"] = score2
        wandb_data["overall_score/diff"] = score2 - score1
    
    # 根据综合分数决定方向
    # 如果第一次高于第二次，计算第一次相对于第二次的最佳改进帧（reverse=True）
    # 如果第二次高于第一次，计算第二次相对于第一次的最佳改进帧（reverse=False）
    reverse = score1 > score2
    
    if reverse:
        logger.info(f"{dir1_name} 评估结果更好（{score1:.6f} > {score2:.6f}），将计算 {dir1_name} 相对于 {dir2_name} 的最佳改进帧")
    else:
        logger.info(f"{dir2_name} 评估结果更好（{score2:.6f} > {score1:.6f}），将计算 {dir2_name} 相对于 {dir1_name} 的最佳改进帧")
    
    # 获取每个相机的最佳改进帧（根据综合分数决定方向）
    best_frames_per_cam = {}
    if len(common_cam_ids) > 0:
        logger.info(f"\n开始查找每个相机的最佳改进帧...")
        best_frames_per_cam = find_best_improved_frames_per_camera(
            args.log_dir1,
            args.log_dir2,
            common_cam_ids=common_cam_ids,
            metrics1=metrics1,
            metrics2=metrics2,
            reverse=reverse
        )
        logger.info(f"找到 {len(best_frames_per_cam)} 个相机的最佳改进帧")
        
        # 输出最佳改进帧信息到控制台
        if best_frames_per_cam:
            logger.info(f"\n{'='*80}")
            if reverse:
                logger.info(f"各相机最佳改进帧信息（{dir1_name} 相对于 {dir2_name} 提升最大的帧）")
            else:
                logger.info(f"各相机最佳改进帧信息（{dir2_name} 相对于 {dir1_name} 提升最大的帧）")
            logger.info(f"{'='*80}")
            for cam_id in sorted(best_frames_per_cam.keys()):
                best_frame = best_frames_per_cam[cam_id]
                logger.info(f"\n相机 {cam_id}:")
                logger.info(f"  最佳改进帧ID: {best_frame.get('frame_idx', 'N/A')}")
                logger.info(f"  改进分数: {best_frame.get('improvement_score', 0):.6f}")
                logger.info(f"  {dir1_name} 图像路径: {best_frame.get('image_path_old', 'N/A')}")
                logger.info(f"  {dir2_name} 图像路径: {best_frame.get('image_path_new', 'N/A')}")
    else:
        logger.warning("没有共同的相机ID，无法查找最佳改进帧")
    
    for metric_name, metric_data in sorted_improved:
        logger.info(f"\n{metric_name}:")
        logger.info(f"  {dir1_name}: {metric_data['old']:.6f}")
        logger.info(f"  {dir2_name}: {metric_data['new']:.6f}")
        logger.info(f"  变化: {metric_data['diff']:+.6f} ({metric_data['improvement_pct']:+.2f}%)")
        
        # 记录到 wandb
        if args.enable_wandb:
            wandb_data[f"comparison/{metric_name}/old"] = metric_data['old']
            wandb_data[f"comparison/{metric_name}/new"] = metric_data['new']
            wandb_data[f"comparison/{metric_name}/diff"] = metric_data['diff']
            wandb_data[f"comparison/{metric_name}/improvement_pct"] = metric_data['improvement_pct']
    
    # 为所有相机输出对比图片到 wandb（只输出一次，不按指标重复）
    if args.enable_wandb and best_frames_per_cam:
        logger.info(f"\n开始上传最佳改进帧图片到 wandb...")
        for cam_id, best_frame in best_frames_per_cam.items():
            if best_frame.get("image_path_old") and best_frame.get("image_path_new"):
                if os.path.exists(best_frame["image_path_old"]) and os.path.exists(best_frame["image_path_new"]):
                    try:
                        img_old = imageio.imread(best_frame["image_path_old"])
                        img_new = imageio.imread(best_frame["image_path_new"])
                        
                        frame_idx = best_frame.get('frame_idx', 'N/A')
                        
                        # 上传图像，使用文件夹名称
                        wandb_images[f"best_improvement/cam{cam_id}_{dir1_name}"] = wandb.Image(
                            img_old,
                            caption=f"Cam {cam_id}, Frame {frame_idx} - {dir1_name}"
                        )
                        
                        # 上传新图像
                        wandb_images[f"best_improvement/cam{cam_id}_{dir2_name}"] = wandb.Image(
                            img_new,
                            caption=f"Cam {cam_id}, Frame {frame_idx} - {dir2_name}"
                        )
                        
                        logger.info(f"  已上传相机 {cam_id} 的最佳改进帧 (Frame {frame_idx})")
                    except Exception as e:
                        logger.warning(f"Error loading images for cam {cam_id}: {e}")
    
    # 记录所有对比结果到 wandb
    if args.enable_wandb:
        logger.info(f"\n{'='*80}")
        logger.info("开始上传结果到 wandb...")
        
        # 上传指标数据
        logger.info(f"  上传指标数据: {len(wandb_data)} 个数据点")
        wandb.log(wandb_data)
        
        # 上传图片
        if wandb_images:
            logger.info(f"  上传对比图片: {len(wandb_images)} 张图片")
            wandb.log(wandb_images)
        
        # 创建对比表格（只包含全图指标）
        table_data = []
        for metric_name, metric_data in full_comparison.items():
            table_data.append([
                metric_name,
                f"{metric_data['old']:.6f}",
                f"{metric_data['new']:.6f}",
                f"{metric_data['diff']:+.6f}",
                f"{metric_data['improvement_pct']:+.2f}%",
                "✓" if metric_data['improved'] else "✗"
            ])
        
        table = wandb.Table(
            columns=["Metric", dir1_name, dir2_name, "Difference", "Improvement %", "Improved"],
            data=table_data
        )
        logger.info(f"  上传指标对比表格: {len(table_data)} 行数据")
        wandb.log({"comparison_table": table})
        
        # 创建最佳改进帧信息表格（只包含图片）
        if best_frames_per_cam:
            best_frames_table_data = []
            for cam_id in sorted(best_frames_per_cam.keys()):
                best_frame = best_frames_per_cam[cam_id]
                
                # 加载图片用于表格显示
                img_old_cell = None
                img_new_cell = None
                
                old_path = best_frame.get('image_path_old', '')
                new_path = best_frame.get('image_path_new', '')
                
                if old_path and os.path.exists(old_path):
                    try:
                        img_old = imageio.imread(old_path)
                        img_old_cell = wandb.Image(img_old)
                    except Exception as e:
                        logger.warning(f"无法加载旧图像用于表格 (Cam {cam_id}): {e}")
                        img_old_cell = os.path.basename(old_path) if old_path else 'N/A'
                
                if new_path and os.path.exists(new_path):
                    try:
                        img_new = imageio.imread(new_path)
                        img_new_cell = wandb.Image(img_new)
                    except Exception as e:
                        logger.warning(f"无法加载新图像用于表格 (Cam {cam_id}): {e}")
                        img_new_cell = os.path.basename(new_path) if new_path else 'N/A'
                
                # 如果图片加载失败，使用文件名作为后备
                if img_old_cell is None:
                    img_old_cell = os.path.basename(old_path) if old_path else 'N/A'
                if img_new_cell is None:
                    img_new_cell = os.path.basename(new_path) if new_path else 'N/A'
                
                best_frames_table_data.append([
                    f"Cam {cam_id}",
                    best_frame.get('frame_idx', 'N/A'),
                    f"{best_frame.get('improvement_score', 0):.6f}",
                    img_old_cell,
                    img_new_cell
                ])
            
            best_frames_table = wandb.Table(
                columns=["Camera ID", "Frame ID", "Improvement Score", dir1_name, dir2_name],
                data=best_frames_table_data
            )
            logger.info(f"  上传最佳改进帧表格（包含图片）: {len(best_frames_table_data)} 行数据")
            wandb.log({"best_improvement_frames_table": best_frames_table})
        
        # 保存表格到本地文件
        if best_frames_per_cam:
            try:
                # 提取模型名称
                old_model_name = os.path.basename(args.log_dir1.rstrip('/'))
                new_model_name = os.path.basename(args.log_dir2.rstrip('/'))
                
                # 构建保存路径：从 log_dir1 中提取基础路径
                # log_dir1 格式: output/qcraft_20251025_163358_QCOYSD504206_1595_1610/old_model
                # 需要提取: output/qcraft_20251025_163358_QCOYSD504206_1595_1610
                # 转换为绝对路径
                log_dir1_abs = os.path.abspath(args.log_dir1)
                base_output_dir = os.path.dirname(log_dir1_abs)
                
                # 构建完整保存路径
                save_base_dir = os.path.join(base_output_dir, "compare_model")
                os.makedirs(save_base_dir, exist_ok=True)
                
                # 保存最佳改进帧表格
                filename = f"best_improvement_frames_table_{old_model_name}_{new_model_name}.csv"
                filepath = os.path.join(save_base_dir, filename)
                
                # 保存为 CSV 文件（图片路径）
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # 写入表头
                    writer.writerow(["Camera ID", "Frame ID", "Improvement Score", f"{dir1_name} Image Path", f"{dir2_name} Image Path"])
                    # 写入数据（使用实际图片路径而不是 wandb.Image 对象）
                    for cam_id in sorted(best_frames_per_cam.keys()):
                        best_frame = best_frames_per_cam[cam_id]
                        writer.writerow([
                            f"Cam {cam_id}",
                            best_frame.get('frame_idx', 'N/A'),
                            f"{best_frame.get('improvement_score', 0):.6f}",
                            best_frame.get('image_path_old', 'N/A'),
                            best_frame.get('image_path_new', 'N/A')
                        ])
                
                logger.info(f"  已保存最佳改进帧表格到本地: {filepath}")
                
                # 同时保存图片到本地
                images_dir = os.path.join(save_base_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                
                # 保存最佳改进帧图片
                saved_images = 0
                for cam_id in sorted(best_frames_per_cam.keys()):
                    best_frame = best_frames_per_cam[cam_id]
                    frame_idx = best_frame.get('frame_idx', 'N/A')
                    old_path = best_frame.get('image_path_old', '')
                    new_path = best_frame.get('image_path_new', '')
                    
                    # 保存图像，使用文件夹名称
                    if old_path and os.path.exists(old_path):
                        try:
                            img_old = imageio.imread(old_path)
                            old_save_path = os.path.join(images_dir, f"cam{cam_id}_frame{frame_idx}_{dir1_name}.png")
                            imageio.imwrite(old_save_path, img_old)
                            saved_images += 1
                        except Exception as e:
                            logger.warning(f"无法保存 {dir1_name} 图像 (Cam {cam_id}): {e}")
                    
                    # 保存新图像
                    if new_path and os.path.exists(new_path):
                        try:
                            img_new = imageio.imread(new_path)
                            new_save_path = os.path.join(images_dir, f"cam{cam_id}_frame{frame_idx}_{dir2_name}.png")
                            imageio.imwrite(new_save_path, img_new)
                            saved_images += 1
                        except Exception as e:
                            logger.warning(f"无法保存 {dir2_name} 图像 (Cam {cam_id}): {e}")
                
                if saved_images > 0:
                    logger.info(f"  已保存 {saved_images} 张最佳改进帧图片到: {images_dir}")
                
            except Exception as e:
                logger.error(f"保存表格到本地时出错: {e}", exc_info=True)
        
        logger.info("所有数据已上传到 wandb")
    
    logger.info(f"\n{'='*80}")
    logger.info("对比完成！")
    if args.enable_wandb:
        logger.info(f"结果已上传到 wandb: {wandb.run.url}")
    
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compare two training evaluation results")
    
    # 输入路径
    parser.add_argument("--log_dir1", type=str, required=True, help="First training log directory")
    parser.add_argument("--log_dir2", type=str, required=True, help="Second training log directory")
    
    # wandb 配置
    parser.add_argument("--enable_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    
    # 其他参数
    parser.add_argument("--num_cams", type=int, default=11, help="Number of cameras")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main(args)

