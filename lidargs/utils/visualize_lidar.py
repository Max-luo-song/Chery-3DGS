"""
加载并可视化 LiDAR 点云数据
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_lidar_txt(txt_path):
    """
    输入txt格式的激光雷达点云数据，每行x y z intensity，进行可视化
    """
    # 加载数据
    points = np.loadtxt(txt_path)
    if points.shape[1] < 4:
        raise ValueError("每行必须包含x y z intensity四个值")
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(x, y, z, c=intensity, cmap='viridis', s=1)
    fig.colorbar(p, ax=ax, label='Intensity')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('LiDAR Point Cloud Visualization')
    plt.show()

# 示例用法：
# visualize_lidar_txt('your_lidar_data.txt')

