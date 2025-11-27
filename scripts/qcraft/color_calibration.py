import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import json

class MultiDatasetColorCalibration:
    def __init__(self):
        self.reference_stats = None
        self.calibration_models = {}
        self.reference_dataset = None

    def compute_reference_statistics(self, images: List[np.ndarray],
                                    method: str = 'mean') -> Dict:
        """计算参考颜色统计量"""
        if not images:
            raise ValueError("No images provided")

        all_stats = []

        for i, img in enumerate(images):
            stats_dict = self._compute_image_statistics(img)
            all_stats.append(stats_dict)
            if i < 5:  # 只打印前5张的统计信息
                print(f"Image {i} - Mean RGB: {stats_dict['mean_rgb']}")

        # 计算参考统计量
        if method == 'mean':
            reference = {
                'mean_rgb': np.mean([s['mean_rgb'] for s in all_stats], axis=0),
                'std_rgb': np.mean([s['std_rgb'] for s in all_stats], axis=0),
                'histograms': None
            }
        elif method == 'median':
            reference = {
                'mean_rgb': np.median([s['mean_rgb'] for s in all_stats], axis=0),
                'std_rgb': np.median([s['std_rgb'] for s in all_stats], axis=0),
                'histograms': None
            }
        elif method == 'robust':
            all_means = np.array([s['mean_rgb'] for s in all_stats])
            all_stds = np.array([s['std_rgb'] for s in all_stats])

            reference = {
                'mean_rgb': np.median(all_means, axis=0),
                'std_rgb': np.median(all_stds, axis=0),
                'histograms': None
            }

        self.reference_stats = reference
        print(f"\nReference RGB Mean: {reference['mean_rgb']}")
        print(f"Reference RGB Std: {reference['std_rgb']}")

        return reference

    def _compute_image_statistics(self, img: np.ndarray) -> Dict:
        """计算单张图像的统计量"""
        if len(img.shape) == 3:
            mean_rgb = np.mean(img, axis=(0, 1))
            std_rgb = np.std(img, axis=(0, 1))

            histograms = []
            for i in range(img.shape[2]):
                hist, _ = np.histogram(img[:, :, i], bins=256, range=(0, 256))
                histograms.append(hist)
        else:
            mean_rgb = np.array([np.mean(img)])
            std_rgb = np.array([np.std(img)])
            hist, _ = np.histogram(img, bins=256, range=(0, 256))
            histograms = [hist]

        return {
            'mean_rgb': mean_rgb,
            'std_rgb': std_rgb,
            'histograms': histograms
        }

    def calibrate_all_images(self, images: List[np.ndarray],
                             reference_method: str = 'median',
                             calibration_method: str = 'statistical') -> List[np.ndarray]:
        """校准所有图像到统一的颜色标准"""
        print("=== 开始图像颜色校准 ===")

        # 1. 计算参考标准
        self.compute_reference_statistics(images, reference_method)

        # 2. 校准每张图像
        calibrated_images = []

        for i, img in enumerate(images):
            if i % 100 == 0:
                print(f"\n校准图像 {i}/{len(images)}...")

            if calibration_method == 'statistical':
                calibrated = self._statistical_calibration(img)
            elif calibration_method == 'linear':
                calibrated = self._linear_calibration(img)
            elif calibration_method == 'histogram':
                calibrated = self._histogram_calibration(img)
            else:
                raise ValueError("Unknown calibration method")

            calibrated_images.append(calibrated)

            # 评估校准效果
            if i < 5:
                original_stats = self._compute_image_statistics(img)
                calibrated_stats = self._compute_image_statistics(calibrated)

                print(f"  原始均值: {original_stats['mean_rgb']}")
                print(f"  校准后均值: {calibrated_stats['mean_rgb']}")
                print(f"  与参考差异: {np.abs(calibrated_stats['mean_rgb'] - self.reference_stats['mean_rgb'])}")

        print("\n=== 校准完成 ===")
        return calibrated_images

    def _statistical_calibration(self, img: np.ndarray) -> np.ndarray:
        """基于统计量的校准"""
        img_stats = self._compute_image_statistics(img)

        if len(img.shape) == 3:
            calibrated = np.zeros_like(img, dtype=np.float32)
            for i in range(img.shape[2]):
                channel = img[:, :, i].astype(np.float32)

                if img_stats['std_rgb'][i] > 0:
                    normalized = (channel - img_stats['mean_rgb'][i]) / img_stats['std_rgb'][i]
                    calibrated[:, :, i] = normalized * self.reference_stats['std_rgb'][i] + self.reference_stats['mean_rgb'][i]
                else:
                    calibrated[:, :, i] = channel

            return np.clip(calibrated, 0, 255).astype(np.uint8)
        else:
            channel = img.astype(np.float32)
            if img_stats['std_rgb'][0] > 0:
                normalized = (channel - img_stats['mean_rgb'][0]) / img_stats['std_rgb'][0]
                calibrated = normalized * self.reference_stats['std_rgb'][0] + self.reference_stats['mean_rgb'][0]
            else:
                calibrated = channel

            return np.clip(calibrated, 0, 255).astype(np.uint8)

    def _linear_calibration(self, img: np.ndarray) -> np.ndarray:
        """线性校准方法"""
        img_stats = self._compute_image_statistics(img)

        if len(img.shape) == 3:
            calibrated = np.zeros_like(img, dtype=np.float32)
            for i in range(img.shape[2]):
                channel = img[:, :, i].astype(np.float32)

                if img_stats['std_rgb'][i] > 0:
                    scale = self.reference_stats['std_rgb'][i] / img_stats['std_rgb'][i]
                    shift = self.reference_stats['mean_rgb'][i] - scale * img_stats['mean_rgb'][i]
                    calibrated[:, :, i] = scale * channel + shift
                else:
                    calibrated[:, :, i] = channel

            return np.clip(calibrated, 0, 255).astype(np.uint8)
        else:
            channel = img.astype(np.float32)
            if img_stats['std_rgb'][0] > 0:
                scale = self.reference_stats['std_rgb'][0] / img_stats['std_rgb'][0]
                shift = self.reference_stats['mean_rgb'][0] - scale * img_stats['mean_rgb'][0]
                calibrated = scale * channel + shift
            else:
                calibrated = channel

            return np.clip(calibrated, 0, 255).astype(np.uint8)

    def _histogram_calibration(self, img: np.ndarray) -> np.ndarray:
        """直方图匹配校准"""
        if len(img.shape) == 3:
            calibrated = np.zeros_like(img)
            for i in range(img.shape[2]):
                ref_mean = self.reference_stats['mean_rgb'][i]
                ref_std = self.reference_stats['std_rgb'][i]

                ref_values = np.random.normal(ref_mean, ref_std, 10000)
                ref_values = np.clip(ref_values, 0, 255).astype(np.uint8)

                calibrated[:, :, i] = self._match_histogram_channel(
                    img[:, :, i], ref_values
                )
            return calibrated
        else:
            ref_mean = self.reference_stats['mean_rgb'][0]
            ref_std = self.reference_stats['std_rgb'][0]
            ref_values = np.random.normal(ref_mean, ref_std, 10000)
            ref_values = np.clip(ref_values, 0, 255).astype(np.uint8)

            return self._match_histogram_channel(img, ref_values)

    def _match_histogram_channel(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """单通道直方图匹配"""
        source_values, source_indices, source_counts = np.unique(
            source, return_inverse=True, return_counts=True
        )
        ref_values, ref_counts = np.unique(reference, return_counts=True)

        source_quantiles = np.cumsum(source_counts).astype(np.float64)
        source_quantiles /= source_quantiles[-1]

        ref_quantiles = np.cumsum(ref_counts).astype(np.float64)
        ref_quantiles /= ref_quantiles[-1]

        interp_ref_values = np.interp(source_quantiles, ref_quantiles, ref_values)
        return interp_ref_values[source_indices].reshape(source.shape)

def load_images_from_directory(image_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """从目录加载图像"""
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    # 查找所有jpg文件
    image_files = sorted([f for f in image_dir.glob("*.jpg")])

    if not image_files:
        raise FileNotFoundError(f"No JPG files found in {image_dir}")

    images = []
    filenames = []

    print(f"Found {len(image_files)} images:")
    for i, img_file in enumerate(image_files[:5]):  # 只打印前5张
        print(f"  Loading: {img_file.name}")
        img = cv2.imread(str(img_file))
        if img is not None:
            images.append(img)
            filenames.append(img_file.name)
        else:
            print(f"  Warning: Could not load {img_file.name}")
    if len(image_files) > 5:
        print(f"  ... and {len(image_files) - 5} more images")

    # 继续加载其余图像
    for img_file in image_files[5:]:
        img = cv2.imread(str(img_file))
        if img is not None:
            images.append(img)
            filenames.append(img_file.name)

    return images, filenames

def process_qcraft_images(image_dir: str, output_dir: str = None,
                         reference_method: str = 'median',
                         calibration_method: str = 'statistical',
                         backup: bool = True):
    """处理QCraft图像的主函数"""

    # 如果未指定输出目录，在原目录创建backup后直接覆盖
    if output_dir is None:
        image_path = Path(image_dir)
        if backup:
            backup_dir = image_path.parent / f"{image_path.name}_backup"
            print(f"\n创建备份目录: {backup_dir}")
            backup_dir.mkdir(exist_ok=True)

    # 创建输出目录（如果指定了的话）
    else:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. 加载所有图像
        print("=== 加载图像 ===")
        images, filenames = load_images_from_directory(image_dir)

        if len(images) == 0:
            print("No images loaded!")
            return

        print(f"\n成功加载 {len(images)} 张图像")

        # 2. 如果备份，先备份原图像
        if backup and output_dir is None:
            print(f"\n=== 备份原始图像 ===")
            for filename in filenames[:5]:
                src_path = Path(image_dir) / filename
                dst_path = backup_dir / filename
                if not dst_path.exists():
                    import shutil
                    shutil.copy2(src_path, dst_path)
            if len(filenames) > 5:
                print(f"  备份剩余 {len(filenames) - 5} 张图像...")
                for filename in filenames[5:]:
                    src_path = Path(image_dir) / filename
                    dst_path = backup_dir / filename
                    if not dst_path.exists():
                        import shutil
                        shutil.copy2(src_path, dst_path)
            print(f"备份完成: {backup_dir}")

        # 3. 初始化校准器
        calibrator = MultiDatasetColorCalibration()

        # 4. 执行校准
        print("\n=== 开始颜色校准 ===")
        calibrated_images = calibrator.calibrate_all_images(
            images,
            reference_method=reference_method,
            calibration_method=calibration_method
        )

        # 5. 保存校准后的图像
        print("\n=== 保存校准后的图像 ===")
        for i, (calibrated_img, filename) in enumerate(zip(calibrated_images, filenames)):
            if output_dir is None:
                # 直接覆盖原图像
                output_path = Path(image_dir) / filename
            else:
                # 保存到输出目录
                output_path = Path(output_dir) / f"calibrated_{filename}"

            cv2.imwrite(str(output_path), calibrated_img)

            if i < 5 or (i + 1) == len(images):
                print(f"Saved: {output_path}")

        # 6. 保存校准信息
        calibration_info = {
            'reference_stats': {
                'mean_rgb': calibrator.reference_stats['mean_rgb'].tolist(),
                'std_rgb': calibrator.reference_stats['std_rgb'].tolist()
            },
            'processed_images': len(images),
            'calibration_method': calibration_method,
            'reference_method': reference_method
        }

        if output_dir is None:
            info_path = Path(image_dir).parent / "calibration_info.json"
        else:
            info_path = Path(output_dir) / "calibration_info.json"

        with open(info_path, 'w') as f:
            json.dump(calibration_info, f, indent=2)

        print(f"\n校准信息保存至: {info_path}")
        print(f"\n=== 处理完成 ===")
        print(f"原始图像目录: {image_dir}")
        if output_dir is None:
            print(f"图像已覆盖，备份在: {backup_dir}")
        else:
            print(f"校准后图像目录: {output_dir}")
        print(f"处理图像数量: {len(images)}")
        print(f"\n色差校准目标已达成！所有图像使用统一的参考标准。")

    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        import traceback
        traceback.print_exc()

def quick_test():
    """快速测试校准功能"""
    print("=== 快速测试 ===")

    # 创建测试数据
    test_images = []
    for i in range(4):
        # 创建不同颜色偏移的测试图像
        base = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        shift = np.array([1.0 + i*0.1, 1.0 - i*0.05, 1.0 + i*0.08])
        img = np.clip(base.astype(np.float32) * shift, 0, 255).astype(np.uint8)
        test_images.append(img)

    # 测试校准
    calibrator = MultiDatasetColorCalibration()
    calibrated = calibrator.calibrate_all_images(test_images)

    # 计算校准前后色差
    original_diffs = []
    calibrated_diffs = []
    for i in range(len(test_images)):
        orig_stats = calibrator._compute_image_statistics(test_images[i])
        calib_stats = calibrator._compute_image_statistics(calibrated[i])

        original_diffs.append(np.linalg.norm(orig_stats['mean_rgb'] - calibrator.reference_stats['mean_rgb']))
        calibrated_diffs.append(np.linalg.norm(calib_stats['mean_rgb'] - calibrator.reference_stats['mean_rgb']))

    print(f"\n原始平均色差: {np.mean(original_diffs):.2f}")
    print(f"校准后平均色差: {np.mean(calibrated_diffs):.2f}")
    print(f"色差降低: {(1 - np.mean(calibrated_diffs) / np.mean(original_diffs)) * 100:.1f}%")

    return test_images, calibrated

if __name__ == "__main__":
    # 目标图像目录
    target_dir = "data/qcraft/processed/training/20251025_163358_QCOYSD504206/images"

    # 检查目录是否存在
    if not os.path.exists(target_dir):
        print(f"目标目录不存在: {target_dir}")
        print("请确认目录路径是否正确")
    else:
        print(f"开始处理目录: {target_dir}")

        # 执行图像校准
        # 参数说明：
        # - image_dir: 图像目录
        # - output_dir: 输出目录，如果为None则直接覆盖原图像（会先备份）
        # - reference_method: 参考标准计算方法 ('mean', 'median', 'robust')
        # - calibration_method: 校准方法 ('statistical', 'linear', 'histogram')
        # - backup: 是否备份原图像（仅在直接覆盖时有效）

        process_qcraft_images(
            image_dir=target_dir,
            output_dir=None,  # None 表示直接覆盖原图像（会创建备份）
            reference_method='median',  # 使用中位数作为参考，更鲁棒
            calibration_method='statistical',  # 使用统计方法校准
            backup=True  # 校准前先备份原图像
        )

    # 如果想测试功能，可以取消下面的注释
    # print("\n" + "="*60)
    # quick_test()