"""
解析点云bin文件
格式: x y z intensity timestamp ring lidar_id
"""

import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def read_lidar_bin(bin_path):
    """
    读取bin格式的激光雷达点云数据，每个点包含x, y, z, intensity, timestamp, ring, lidar_id
    二进制文件中每个点包含 7 个 float32 字段：
            x y z intensity timestamp ring lidar_id
    """
    points = []
    with open(bin_path, 'rb') as f:
        byte = f.read(28)  # 每个点占28字节
        while byte:
            point = struct.unpack('fffffff', byte)
            points.append(point)
            byte = f.read(28)
    return np.array(points)

def convert_to_txt(bin_path, txt_path):
    """
    将bin格式的点云数据转换为txt格式
    """
    print(f"Converting {bin_path} to {txt_path}...")
    points = read_lidar_bin(bin_path)
    # 过滤掉强度为0的点
    points = points[points[:, 3] > 0]
    # 过滤掉lidar_id不为0的点
    points = points[points[:, 6] == 0]
    np.savetxt(txt_path, points, fmt='%f %f %f %f %f %d %d')
    print(f"Converted {bin_path} to {txt_path}")

if __name__ == "__main__":
    print("Point Cloud Bin to Txt Converter")
    parser = argparse.ArgumentParser(description="Convert LiDAR bin files to txt format")
    parser.add_argument("--input_bin", type=str, required=True, help="Path to input bin file")
    parser.add_argument("--output_txt", type=str, required=True, help="Path to output txt file")
    args = parser.parse_args()
    print(args)

    convert_to_txt(args.input_bin, args.output_txt)

    # # 可视化点云
    # points = np.loadtxt(args.output_txt)
    # x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # p = ax.scatter(x, y, z, c=intensity, cmap='viridis', s=1)
    # fig.colorbar(p, ax=ax, label='Intensity')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('LiDAR Point Cloud Visualization')
    # plt.show()