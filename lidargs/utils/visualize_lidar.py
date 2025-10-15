import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# ================== 配置 ==================
base_dir = "/home/not0513/data/orinY/processed/test/20250702_133223_Q2517/render_point_4000"
output_root = "/home/not0513/data/orinY/processed/test/20250702_133223_Q2517/render_point_4000_all"

render_dir = os.path.join(output_root, "render")
gt_dir = os.path.join(output_root, "gt")
video_dir = os.path.join(output_root, "video")

os.makedirs(render_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# 视频参数
fps = 5
width, height = 1200, 900
point_size = 0.1
use_intensity = True
# =========================================

def copy_all_pointclouds():
    """将所有 group 的 render/*.txt 和 gt/*.txt 复制到统一目录"""
    copied_render = 0
    copied_gt = 0

    # 遍历所有 group 目录 (0, 1, 2, ...)
    for item in os.listdir(base_dir):
        group_path = os.path.join(base_dir, item)
        if not (os.path.isdir(group_path) and item.isdigit()):
            continue

        render_src = os.path.join(group_path, "render")
        gt_src = os.path.join(group_path, "gt")

        # 复制 render
        if os.path.exists(render_src):
            for f in os.listdir(render_src):
                if f.endswith('.txt'):
                    src_file = os.path.join(render_src, f)
                    dst_file = os.path.join(render_dir, f)
                    # 如果目标已存在，跳过（你保证无冲突，所以通常不会发生）
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
                        copied_render += 1
                    else:
                        print(f"Warning: {dst_file} already exists, skipped.")

        # 复制 gt
        if os.path.exists(gt_src):
            for f in os.listdir(gt_src):
                if f.endswith('.txt'):
                    src_file = os.path.join(gt_src, f)
                    dst_file = os.path.join(gt_dir, f)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
                        copied_gt += 1
                    else:
                        print(f"Warning: {dst_file} already exists, skipped.")

    print(f"Copied {copied_render} render files and {copied_gt} gt files.")
    return copied_render > 0 or copied_gt > 0

def get_sorted_timestamps(folder):
    """按时间戳排序所有 .txt 文件"""
    files = [f for f in os.listdir(folder) if f.endswith('.txt')]
    timestamps = []
    for f in files:
        try:
            ts = int(f.split('_')[0])  # e.g., "100_render_points.txt" -> 100
            timestamps.append(ts)
        except:
            continue
    return sorted(set(timestamps))

def create_top_view_video(data_dir, video_name, fixed_ylim):
    """生成俯视视频，使用 CloudCompare 风格（Z 高度着色）"""
    timestamps = get_sorted_timestamps(data_dir)
    if not timestamps:
        print(f"No valid files in {data_dir}, skip video.")
        return

    video_path = os.path.join(video_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    z_min, z_max = -2.0, 30.0  # 默认范围（可根据场景调整）
    for ts in timestamps:
        # 尝试 render 或 gt 命名
        candidates = [
            f"{ts}_render_points.txt",
            f"{ts}_gt__points.txt"
        ]
        file_path = None
        for cand in candidates:
            fp = os.path.join(data_dir, cand)
            if os.path.exists(fp):
                file_path = fp
                break

        if not file_path:
            continue

        try:
            data = np.loadtxt(file_path, skiprows=1)
            if data.size == 0:
                print(f"Skip frame {ts}: empty")
                continue
            if data.ndim == 1:
                data = data.reshape(1, -1)
            xyz = data[:, :3]
        except Exception as e:
            print(f"Skip frame {ts}: {e}")
            continue

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        # === 每帧动态计算 xlim（少留白）===
        if len(x) > 0:
            x_min, x_max = x.min(), x.max()
            margin_x = 2.0  # 左右各留 2 米
            xlim = (x_min - margin_x, x_max + margin_x)
        else:
            xlim = (-10, 10)

        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.xlim(xlim)
        plt.ylim(fixed_ylim)
        plt.axis('off')

        # === 用 Z 坐标着色 ===
        colors = np.clip((z - z_min) / (z_max - z_min + 1e-6), 0, 1)
        plt.scatter(x, y, c=colors, s=point_size, cmap='jet', vmin=0, vmax=1)

        plt.tight_layout(pad=0)
        plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        frame = cv2.imread('temp_frame.png')
        if frame is not None:
            frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)

    video_writer.release()
    if os.path.exists('temp_frame.png'):
        os.remove('temp_frame.png')

if __name__ == "__main__":
    print("Copying all render/gt point clouds...")
    has_data = copy_all_pointclouds()

    ylim = [-40, 30]

    print("Generating render video...")
    create_top_view_video(render_dir, "render_video.mp4", fixed_ylim=ylim)

    print("Generating GT video...")
    create_top_view_video(gt_dir, "gt_video.mp4", fixed_ylim=ylim)

    print(f"   Done! Output:")
    print(f"   Render: {render_dir}")
    print(f"   GT:     {gt_dir}")
    print(f"   Video:  {video_dir}")
