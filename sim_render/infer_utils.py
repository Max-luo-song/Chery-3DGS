import os
import re
import json
import numpy as np


def find_min_frame_txt(dir_path):
    """
    在目录下查找编号最小的 txt 文件。
    文件名可以是 000.txt、001.txt、0032.txt、12.txt 等任意位数。
    返回最小编号文件的完整路径。
    """
    txt_files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
    if not txt_files:
        raise FileNotFoundError("目录中没有 txt 文件")

    # 从文件名提取数字部分，例如 '0032.txt' -> 32
    def extract_number(name):
        match = re.search(r'(\d+)\.txt$', name)
        return int(match.group(1)) if match else float('inf')

    # 找最小编号的文件
    min_file = min(txt_files, key=extract_number)

    return os.path.join(dir_path, min_file)


def load_transform_matrix(txt_path):
    """
    从 txt 文件中读取 4x4 的齐次变换矩阵。
    文件要求包含 4 行，每行 4 个浮点数。
    """
    matrix = np.loadtxt(txt_path)
    if matrix.shape != (4, 4):
        raise ValueError(f"矩阵尺寸应为 4x4，但读取到的是 {matrix.shape}")
    return matrix


def extract_lidar_extrinsics(json_path):
    """
    从包含 lidar_params -> installation -> extrinsics 的 JSON 文件中
    提取 x, y, z, yaw, pitch, roll。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 假设 lidar_params 是列表，取第一个 lidar 的 extrinsics
    params = data["lidar_params"][0]["installation"]["extrinsics"]

    fields = ["x", "y", "z", "yaw", "pitch", "roll"]
    result = {key: params.get(key) for key in fields}

    return result


