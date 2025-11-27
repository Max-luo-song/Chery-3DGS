#!/usr/bin/env python3
"""
将时间戳文件夹下具有相同前缀的图片按时间戳顺序拼成MP4视频
"""

import os
import re
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm


# 相机名称到索引的映射字典
QCRAFT_CAMERA_DICT = {  # 轻舟
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110": 0,  # 广角前视 FOV110
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60": 1,  # 广角前视 FOV60
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30": 2,  # 长焦前视 FOV30
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15": 3,  # 长焦前视 FOV15
    # "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_LEFT_H60": 4,  # 广角左前 FOV60
    "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99": 4,  # 左前 FOV99
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99": 5,  # 左后 FOV99
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30": 6,  # 左后 FOV30
    # "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_RIGHT_H60": 8,  # 广角右前 FOV60
    "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99": 7,  # 右前 FOV99
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99": 8,  # 右后 FOV99
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30": 9,  # 右后 FOV30
    "CAM_PBQ_REAR_RESET_OPTICAL_H50": 10,  # 后视 FOV50
}


def extract_camera_name(filename):
    """
    从文件名中提取相机名称（前缀）
    例如: 20250702_133223_Q2517-CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99-1751434405.4459.jpg
    返回: CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99
    """
    pattern = r'.*?-(.*?)-\d+\.\d+\.jpg$'
    match = re.match(pattern, filename)
    if match:
        return match.group(1)
    return None


def get_timestamp_folders(base_dir):
    """
    获取所有时间戳文件夹并按时间戳排序
    """
    folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            try:
                # 尝试将文件夹名转换为浮点数（时间戳）
                timestamp = float(item)
                folders.append((timestamp, item_path))
            except ValueError:
                # 如果无法转换为数字，跳过
                continue

    # 按时间戳排序
    folders.sort(key=lambda x: x[0])
    return [folder_path for _, folder_path in folders]


def collect_images_by_camera(base_dir):
    """
    收集所有图片，按相机名称分组
    返回: {camera_name: [(timestamp, image_path), ...]}
    """
    timestamp_folders = get_timestamp_folders(base_dir)
    print(f"找到 {len(timestamp_folders)} 个时间戳文件夹")

    camera_images = defaultdict(list)

    for folder_path in timestamp_folders:
        folder_name = os.path.basename(folder_path)
        try:
            timestamp = float(folder_name)
        except ValueError:
            continue

        # 遍历文件夹中的所有jpg文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                camera_name = extract_camera_name(filename)
                if camera_name:
                    image_path = os.path.join(folder_path, filename)
                    camera_images[camera_name].append((timestamp, image_path))

    # 对每个相机的图片按时间戳排序
    for camera_name in camera_images:
        camera_images[camera_name].sort(key=lambda x: x[0])

    return camera_images


def create_video_from_images(image_paths, output_path, fps=10):
    """
    从图片列表创建视频
    """
    if not image_paths:
        print(f"警告: 没有图片可以创建视频")
        return False

    # 读取第一张图片获取尺寸
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"错误: 无法读取图片 {image_paths[0]}")
        return False

    height, width, _ = first_image.shape

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"正在创建视频: {output_path}")
    print(f"视频尺寸: {width}x{height}, 帧率: {fps}, 总帧数: {len(image_paths)}")

    # 写入所有图片
    for image_path in tqdm(image_paths, desc="处理图片"):
        frame = cv2.imread(image_path)
        if frame is not None:
            # 确保图片尺寸一致
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
        else:
            print(f"警告: 无法读取图片 {image_path}")

    out.release()
    print(f"视频创建完成: {output_path}")
    return True


def main():
    # 配置
    base_dir = "data/qcraft/raw/20251025_163358_QCOYSD504206/20251025_163358_QCOYSD504206_1595_1610"
    output_dir = "video_gt/20251025_163358_QCOYSD504206"
    fps = 10  # 视频帧率，可以根据需要调整

    # 检查基础目录是否存在
    if not os.path.exists(base_dir):
        print(f"错误: 目录 {base_dir} 不存在")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有图片
    print("正在扫描文件夹并收集图片...")
    camera_images = collect_images_by_camera(base_dir)

    print(f"\n找到 {len(camera_images)} 个相机前缀:")
    for camera_name, images in camera_images.items():
        print(f"  {camera_name}: {len(images)} 张图片")

    # 为每个相机创建视频
    print(f"\n开始创建视频...\n")
    for camera_name, images in camera_images.items():
        # 提取图片路径（不需要时间戳）
        image_paths = [img_path for _, img_path in images]

        # 获取相机索引，如果不在字典中则跳过
        if camera_name not in QCRAFT_CAMERA_DICT:
            print(f"警告: 相机 {camera_name} 不在预定义字典中，跳过")
            continue
        
        camera_index = QCRAFT_CAMERA_DICT[camera_name]
        
        # 生成输出文件名，格式化为3位数字，如000.mp4, 001.mp4等
        output_filename = f"{camera_index:03d}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # 创建视频
        create_video_from_images(image_paths, output_path, fps)
        print(f"相机 {camera_name} -> 索引 {camera_index:03d}")  # 添加映射信息
        print()  # 空行分隔

    print("所有视频创建完成！")


if __name__ == "__main__":
    main()
