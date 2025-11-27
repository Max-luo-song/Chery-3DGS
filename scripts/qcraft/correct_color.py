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
    
    def compute_reference_statistics(self, datasets: List[np.ndarray], 
                                    method: str = 'mean') -> Dict:
        """计算参考颜色统计量"""
        if not datasets:
            raise ValueError("No datasets provided")
        
        all_stats = []
        
        for i, data in enumerate(datasets):
            stats_dict = self._compute_dataset_statistics(data)
            all_stats.append(stats_dict)
            print(f"Dataset {i} - Mean RGB: {stats_dict['mean_rgb']}")
        
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
    
    def _compute_dataset_statistics(self, data: np.ndarray) -> Dict:
        """计算单个数据集的统计量"""
        if len(data.shape) == 3:
            mean_rgb = np.mean(data, axis=(0, 1))
            std_rgb = np.std(data, axis=(0, 1))
            
            histograms = []
            for i in range(data.shape[2]):
                hist, _ = np.histogram(data[:, :, i], bins=256, range=(0, 256))
                histograms.append(hist)
        else:
            mean_rgb = np.array([np.mean(data)])
            std_rgb = np.array([np.std(data)])
            hist, _ = np.histogram(data, bins=256, range=(0, 256))
            histograms = [hist]
        
        return {
            'mean_rgb': mean_rgb,
            'std_rgb': std_rgb,
            'histograms': histograms
        }
    
    def calibrate_all_datasets(self, datasets: List[np.ndarray], 
                             reference_method: str = 'mean',
                             calibration_method: str = 'statistical') -> List[np.ndarray]:
        """校准所有数据集到统一的颜色标准"""
        print("=== 开始多数据集颜色校准 ===")
        
        # 1. 计算参考标准
        self.compute_reference_statistics(datasets, reference_method)
        
        # 2. 校准每个数据集
        calibrated_datasets = []
        
        for i, data in enumerate(datasets):
            print(f"\n校准数据集 {i}...")
            
            if calibration_method == 'statistical':
                calibrated = self._statistical_calibration(data)
            elif calibration_method == 'linear':
                calibrated = self._linear_calibration(data)
            elif calibration_method == 'histogram':
                calibrated = self._histogram_calibration(data)
            else:
                raise ValueError("Unknown calibration method")
            
            calibrated_datasets.append(calibrated)
            
            # 评估校准效果
            original_stats = self._compute_dataset_statistics(data)
            calibrated_stats = self._compute_dataset_statistics(calibrated)
            
            print(f"  原始均值: {original_stats['mean_rgb']}")
            print(f"  校准后均值: {calibrated_stats['mean_rgb']}")
            print(f"  与参考差异: {np.abs(calibrated_stats['mean_rgb'] - self.reference_stats['mean_rgb'])}")
        
        print("\n=== 校准完成 ===")
        return calibrated_datasets
    
    def _statistical_calibration(self, data: np.ndarray) -> np.ndarray:
        """基于统计量的校准"""
        data_stats = self._compute_dataset_statistics(data)
        
        if len(data.shape) == 3:
            calibrated = np.zeros_like(data, dtype=np.float32)
            for i in range(data.shape[2]):
                channel = data[:, :, i].astype(np.float32)
                
                if data_stats['std_rgb'][i] > 0:
                    normalized = (channel - data_stats['mean_rgb'][i]) / data_stats['std_rgb'][i]
                    calibrated[:, :, i] = normalized * self.reference_stats['std_rgb'][i] + self.reference_stats['mean_rgb'][i]
                else:
                    calibrated[:, :, i] = channel
            
            return np.clip(calibrated, 0, 255).astype(np.uint8)
        else:
            channel = data.astype(np.float32)
            if data_stats['std_rgb'][0] > 0:
                normalized = (channel - data_stats['mean_rgb'][0]) / data_stats['std_rgb'][0]
                calibrated = normalized * self.reference_stats['std_rgb'][0] + self.reference_stats['mean_rgb'][0]
            else:
                calibrated = channel
            
            return np.clip(calibrated, 0, 255).astype(np.uint8)
    
    def _linear_calibration(self, data: np.ndarray) -> np.ndarray:
        """线性校准方法"""
        data_stats = self._compute_dataset_statistics(data)
        
        if len(data.shape) == 3:
            calibrated = np.zeros_like(data, dtype=np.float32)
            for i in range(data.shape[2]):
                channel = data[:, :, i].astype(np.float32)
                
                if data_stats['std_rgb'][i] > 0:
                    scale = self.reference_stats['std_rgb'][i] / data_stats['std_rgb'][i]
                    shift = self.reference_stats['mean_rgb'][i] - scale * data_stats['mean_rgb'][i]
                    calibrated[:, :, i] = scale * channel + shift
                else:
                    calibrated[:, :, i] = channel
            
            return np.clip(calibrated, 0, 255).astype(np.uint8)
        else:
            channel = data.astype(np.float32)
            if data_stats['std_rgb'][0] > 0:
                scale = self.reference_stats['std_rgb'][0] / data_stats['std_rgb'][0]
                shift = self.reference_stats['mean_rgb'][0] - scale * data_stats['mean_rgb'][0]
                calibrated = scale * channel + shift
            else:
                calibrated = channel
            
            return np.clip(calibrated, 0, 255).astype(np.uint8)
    
    def _histogram_calibration(self, data: np.ndarray) -> np.ndarray:
        """直方图匹配校准"""
        if len(data.shape) == 3:
            calibrated = np.zeros_like(data)
            for i in range(data.shape[2]):
                ref_mean = self.reference_stats['mean_rgb'][i]
                ref_std = self.reference_stats['std_rgb'][i]
                
                ref_values = np.random.normal(ref_mean, ref_std, 10000)
                ref_values = np.clip(ref_values, 0, 255).astype(np.uint8)
                
                calibrated[:, :, i] = self._match_histogram_channel(
                    data[:, :, i], ref_values
                )
            return calibrated
        else:
            ref_mean = self.reference_stats['mean_rgb'][0]
            ref_std = self.reference_stats['std_rgb'][0]
            ref_values = np.random.normal(ref_mean, ref_std, 10000)
            ref_values = np.clip(ref_values, 0, 255).astype(np.uint8)
            
            return self._match_histogram_channel(data, ref_values)
    
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

def analyze_color_difference(images: List[np.ndarray], filenames: List[str]) -> Dict:
    """分析图像间的颜色差异"""
    print("=== 原始图像颜色差异分析 ===")
    
    # 计算每对图像的差异
    differences = []
    for i, (img1, name1) in enumerate(zip(images, filenames)):
        for j, (img2, name2) in enumerate(zip(images, filenames)):
            if i < j:
                # 计算颜色差异
                mean1 = np.mean(img1, axis=(0,1))
                mean2 = np.mean(img2, axis=(0,1))
                diff = np.abs(mean1 - mean2)
                total_diff = np.sum(diff)
                
                print(f"{name1} vs {name2}:")
                print(f"  RGB差异 = {diff}")
                print(f"  总差异 = {total_diff:.2f}")
                
                differences.append({
                    'image1': name1,
                    'image2': name2,
                    'rgb_diff': diff.tolist(),
                    'total_diff': float(total_diff)
                })
    
    # 计算整体统计
    all_means = []
    for img, name in zip(images, filenames):
        mean_rgb = np.mean(img, axis=(0,1))
        all_means.append(mean_rgb)
        print(f"{name}: RGB均值 = {mean_rgb}")
    
    all_means = np.array(all_means)
    mean_std = np.std(all_means, axis=0)
    total_std = np.mean(mean_std)
    
    print(f"\n=== 整体颜色差异统计 ===")
    print(f"RGB通道标准差: {mean_std}")
    print(f"平均颜色差异: {total_std:.2f}")
    
    return {
        'pairwise_differences': differences,
        'overall_std': mean_std.tolist(),
        'mean_difference': float(total_std),
        'individual_means': [m.tolist() for m in all_means]
    }

def analyze_calibration_results(original_images: List[np.ndarray], 
                             calibrated_images: List[np.ndarray], 
                             filenames: List[str]) -> Dict:
    """分析校准前后的颜色差异"""
    print("\n=== 校准效果分析 ===")
    
    # 分析校准前的差异
    original_analysis = analyze_color_difference(original_images, filenames)
    
    # 分析校准后的差异
    print("\n=== 校准后图像颜色差异分析 ===")
    calibrated_differences = []
    for i, (img1, name1) in enumerate(zip(calibrated_images, filenames)):
        for j, (img2, name2) in enumerate(zip(calibrated_images, filenames)):
            if i < j:
                mean1 = np.mean(img1, axis=(0,1))
                mean2 = np.mean(img2, axis=(0,1))
                diff = np.abs(mean1 - mean2)
                total_diff = np.sum(diff)
                
                print(f"Calibrated {name1} vs Calibrated {name2}:")
                print(f"  RGB差异 = {diff}")
                print(f"  总差异 = {total_diff:.2f}")
                
                calibrated_differences.append({
                    'image1': name1,
                    'image2': name2,
                    'rgb_diff': diff.tolist(),
                    'total_diff': float(total_diff)
                })
    
    # 计算校准后的整体统计
    all_calibrated_means = []
    for img, name in zip(calibrated_images, filenames):
        mean_rgb = np.mean(img, axis=(0,1))
        all_calibrated_means.append(mean_rgb)
        print(f"Calibrated {name}: RGB均值 = {mean_rgb}")
    
    all_calibrated_means = np.array(all_calibrated_means)
    calibrated_std = np.std(all_calibrated_means, axis=0)
    calibrated_total_std = np.mean(calibrated_std)
    
    print(f"\n=== 校准后整体颜色差异统计 ===")
    print(f"RGB通道标准差: {calibrated_std}")
    print(f"平均颜色差异: {calibrated_total_std:.2f}")
    
    # 计算改善程度
    original_mean_diff = original_analysis['mean_difference']
    calibrated_mean_diff = calibrated_total_std
    improvement_ratio = original_mean_diff / (calibrated_mean_diff + 1e-6)
    
    print(f"\n=== 改善效果 ===")
    print(f"校准前平均差异: {original_mean_diff:.2f}")
    print(f"校准后平均差异: {calibrated_mean_diff:.2f}")
    print(f"改善倍数: {improvement_ratio:.2f}x")
    
    if improvement_ratio > 1.5:
        print("✓ 色差校准效果显著")
    elif improvement_ratio > 1.1:
        print("○ 色差校准效果一般")
    else:
        print("✗ 色差校准效果不明显")
    
    return {
        'original_analysis': original_analysis,
        'calibrated_analysis': {
            'pairwise_differences': calibrated_differences,
            'overall_std': calibrated_std.tolist(),
            'mean_difference': float(calibrated_total_std),
            'individual_means': [m.tolist() for m in all_calibrated_means]
        },
        'improvement_ratio': float(improvement_ratio),
        'original_mean_diff': float(original_mean_diff),
        'calibrated_mean_diff': float(calibrated_mean_diff)
    }

def create_comparison_visualization(original_images: List[np.ndarray], 
                                 calibrated_images: List[np.ndarray], 
                                 filenames: List[str], 
                                 output_dir: str):
    """创建对比可视化"""
    print("\n=== 创建对比可视化 ===")
    
    # 创建颜色均值对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始图像颜色均值
    original_means = []
    for img in original_images:
        mean_rgb = np.mean(img, axis=(0,1))
        original_means.append(mean_rgb)
    
    original_means = np.array(original_means)
    
    # 校准后图像颜色均值
    calibrated_means = []
    for img in calibrated_images:
        mean_rgb = np.mean(img, axis=(0,1))
        calibrated_means.append(mean_rgb)
    
    calibrated_means = np.array(calibrated_means)
    
    # 绘制原始图像颜色分布
    x = np.arange(len(filenames))
    width = 0.25
    
    ax1.bar(x - width, original_means[:, 0], width, label='R', color='red', alpha=0.7)
    ax1.bar(x, original_means[:, 1], width, label='G', color='green', alpha=0.7)
    ax1.bar(x + width, original_means[:, 2], width, label='B', color='blue', alpha=0.7)
    
    ax1.set_xlabel('Images')
    ax1.set_ylabel('Mean RGB Value')
    ax1.set_title('Original Images - Color Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f[:8] for f in filenames], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制校准后图像颜色分布
    ax2.bar(x - width, calibrated_means[:, 0], width, label='R', color='red', alpha=0.7)
    ax2.bar(x, calibrated_means[:, 1], width, label='G', color='green', alpha=0.7)
    ax2.bar(x + width, calibrated_means[:, 2], width, label='B', color='blue', alpha=0.7)
    
    ax2.set_xlabel('Images')
    ax2.set_ylabel('Mean RGB Value')
    ax2.set_title('Calibrated Images - Color Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f[:8] for f in filenames], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    comparison_path = os.path.join(output_dir, 'color_calibration_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存到: {comparison_path}")
    
    return comparison_path

def load_images_from_directory(image_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """从目录加载图像，只关注0、1、4、7四个视角"""
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    
    # 查找所有jpg文件
    all_image_files = sorted([f for f in image_dir.glob("*.jpg")])
    
    if not all_image_files:
        raise FileNotFoundError(f"No JPG files found in {image_dir}")
    
    # 只保留0、1、4、7四个视角的图像
    target_prefixes = ['_0', '_1', '_4', '_7']
    filtered_files = []
    
    for img_file in all_image_files:
        # 检查文件名是否以目标前缀开头
        for prefix in target_prefixes:
            if img_file.name.ends(prefix):
                filtered_files.append(img_file)
                break
    
    if not filtered_files:
        raise FileNotFoundError(f"No images with prefixes {target_prefixes} found in {image_dir}")
    
    images = []
    filenames = []
    
    print(f"Found {len(filtered_files)} images (0,1,4,7 views only):")
    for img_file in filtered_files:
        print(f"  Loading: {img_file.name}")
        img = cv2.imread(str(img_file))
        if img is not None:
            images.append(img)
            filenames.append(img_file.name)
        else:
            print(f"  Warning: Could not load {img_file.name}")
    
    return images, filenames

def process_qcraft_images():
    """处理QCraft图像的主函数"""
    
    # 图像目录
    image_dir = "data/qcraft/processed/training/20251025_163358_QCOYSD504206/images"
    
    # 输出目录
    output_dir = "data/qcraft/processed/training/20251025_163358_QCOYSD504206/calibrated_images"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 加载所有图像
        print("=== 加载图像 ===")
        images, filenames = load_images_from_directory(image_dir)
        
        if len(images) == 0:
            print("No images loaded!")
            return
        
        print(f"\nSuccessfully loaded {len(images)} images")
        
        # 2. 分析原始图像的颜色差异
        print("\n" + "="*50)
        original_analysis = analyze_color_difference(images, filenames)
        
        # 3. 初始化校准器
        calibrator = MultiDatasetColorCalibration()
        
        # 4. 执行校准
        print("\n" + "="*50)
        print("=== 开始颜色校准 ===")
        calibrated_images = calibrator.calibrate_all_datasets(
            images,
            reference_method='robust',      # 使用鲁棒方法计算参考标准
            calibration_method='statistical' # 使用统计方法进行校准
        )
        
        # 5. 分析校准效果
        print("\n" + "="*50)
        calibration_results = analyze_calibration_results(images, calibrated_images, filenames)
        
        # 6. 创建对比可视化
        print("\n" + "="*50)
        comparison_path = create_comparison_visualization(images, calibrated_images, filenames, output_dir)
        
        # 7. 保存校准后的图像
        print("\n=== 保存校准后的图像 ===")
        for i, (calibrated_img, filename) in enumerate(zip(calibrated_images, filenames)):
            output_path = os.path.join(output_dir, f"calibrated_{filename}")
            cv2.imwrite(output_path, calibrated_img)
            print(f"Saved: {output_path}")
        
        # 8. 保存详细的校准信息
        calibration_info = {
            'reference_stats': {
                'mean_rgb': calibrator.reference_stats['mean_rgb'].tolist(),
                'std_rgb': calibrator.reference_stats['std_rgb'].tolist()
            },
            'processed_images': len(images),
            'output_directory': output_dir,
            'original_analysis': original_analysis,
            'calibration_results': calibration_results,
            'comparison_chart': comparison_path
        }
        
        info_path = os.path.join(output_dir, "calibration_analysis.json")
        with open(info_path, 'w') as f:
            json.dump(calibration_info, f, indent=2)
        
        print(f"\n=== 处理完成 ===")
        print(f"原始图像目录: {image_dir}")
        print(f"校准后图像目录: {output_dir}")
        print(f"处理图像数量: {len(images)}")
        print(f"详细分析报告: {info_path}")
        print(f"对比图表: {comparison_path}")
        
        # 9. 输出总结
        print(f"\n=== 校准总结 ===")
        print(f"原始平均色差: {calibration_results['original_mean_diff']:.2f}")
        print(f"校准后平均色差: {calibration_results['calibrated_mean_diff']:.2f}")
        print(f"改善倍数: {calibration_results['improvement_ratio']:.2f}x")
        
        if calibration_results['improvement_ratio'] > 1.5:
            print("✓ 色差校准效果显著，图像间颜色一致性大幅提升")
        elif calibration_results['improvement_ratio'] > 1.1:
            print("○ 色差校准有一定效果，图像间颜色一致性有所改善")
        else:
            print("✗ 色差校准效果不明显，可能原始图像色差就很小")
        
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        import traceback
        traceback.print_exc()

def quick_test_calibration():
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
    
    test_filenames = [f"test_{i}.jpg" for i in range(4)]
    
    # 分析原始差异
    original_analysis = analyze_color_difference(test_images, test_filenames)
    
    # 测试校准
    calibrator = MultiDatasetColorCalibration()
    calibrated = calibrator.calibrate_all_datasets(test_images)
    
    # 分析校准效果
    calibration_results = analyze_calibration_results(test_images, calibrated, test_filenames)
    
    print("测试完成!")
    return test_images, calibrated, calibration_results

if __name__ == "__main__":
    # 选择运行模式
    mode = "1"
    
    if mode == "1":
        process_qcraft_images()
    elif mode == "2":
        quick_test_calibration()
    else:
        print("无效选择，运行快速测试...")
        quick_test_calibration()
