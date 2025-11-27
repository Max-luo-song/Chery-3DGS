#!/usr/bin/env python3
"""
调试程序 - 分析图片的实际像素值分布
"""
import cv2
import numpy as np

def analyze_pixel_values(image_path):
    """
    分析图像的实际像素值分布
    """
    try:
        # 读取图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 图片尺寸
        height, width = img.shape

        print(f"图片路径: {image_path}")
        print(f"图片尺寸: {height}x{width}")
        print(f"图片类型: {img.dtype}")
        print()

        # 查找图像中的实际像素值范围
        min_val = np.min(img)
        max_val = np.max(img)
        unique_vals = np.unique(img)

        print(f"像素值范围: {min_val} - {max_val}")
        print(f"不重复像素值: {unique_vals}")
        print(f"不重复值的数量: {len(unique_vals)}")
        print()

        # 统像素值在每行的分布
        print("每行像素值统计:")
        for i in range(height):
            row_min = np.min(img[i, :])
            row_max = np.max(img[i, :])
            row_unique = len(np.unique(img[i, :]))
            if i < 5 or i > height - 5:
                # 显示前5行和后5行
                row_values = img[i, 0:10]  # 每行前10个像素
                print(f"行{i+1}: 值范围={row_min}-{row_max}, 不重复值数={row_unique}")
                print(f"  前10个像素值: {row_values}")

        # 检测边
        edges = cv2.Canny(img, 50, 150)
        edge_pixels = np.sum(edges > 0)

        print()
        print(f"检测到的边缘像素数: {edge_pixels}")
        print()

        return img

    except Exception as e:
        print(f"处理图片时发错误: {e}")
        return None

if __name__ == '__main__':
    image_path = '/inspire/hdd/project/continuinglearningtheory/guoluosong-253108120129/data/qcraft/processed/training/20251025_163358_QCOYSD504206/ego_masks/0.png'
    analyze_pixel_values(image_path)