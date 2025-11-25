import cv2
import numpy as np
import torch
import imageio
from PIL import Image
import os
import glob
from typing import Dict, List, Tuple, Generator

device = "cuda"

fps = 10


def load_video(video: str) -> Generator[Image.Image, None, None]:
    """
    Loads a video and yields each frame as a PIL Image.

    Args:
        video (str):
            The path to the video file. Must be a valid local path to a video file.

    Yields:
        PIL.Image.Image:
            A PIL Image for each frame of the video.

    Raises:
        ValueError:
            If the video path is invalid or the file is not a valid video.
    """
    if not isinstance(video, str):
        raise ValueError("Video input must be a string path to a video file.")

    if not os.path.isfile(video):
        raise ValueError(f"Invalid video path: {video} is not a valid file.")

    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image = pil_image.convert("RGB")
            frames.append(np.array(pil_image))
    finally:
        cap.release()

    return frames, fps


def layout_qcraft(imgs: List[np.array], use_gpu=True) -> np.array:
    """Combine cameras into a tiled image.
    Layout:

        ##########################################################################################################
        #                   # front_tele_30         # front_wide_60     # front_tele_15                          #
        # front_left_99     # front_wide_left_60    # front_wide_110    # front_wide_right_60   # front_right_99 #
        # rear_left_30      # rear_left_99          # rear_50           # rear_right_99         # rear_right_30  #
        ##########################################################################################################
    """

    cam_names = [
        "front_wide_110",
        "front_wide_60",
        "front_tele_30",
        "front_tele_15",
        # "front_wide_left_60",
        "front_left_99",
        "rear_left_99",
        "rear_left_30",
        # "front_wide_right_60",
        "front_right_99",
        "rear_right_99",
        "rear_right_30",
        "rear_50",
    ]

    channel = imgs[0].shape[-1]
    if use_gpu:
        imgs = [torch.from_numpy(img).to(device, dtype=torch.float32) for img in imgs]

    max_width = 0
    max_height = 0
    min_width = 1e10
    min_height = 1e10

    for img in imgs:
        max_width = max(max_width, img.shape[1])
        max_height = max(max_height, img.shape[0])
        min_width = min(min_width, img.shape[1])
        min_height = min(min_height, img.shape[0])

    width = 5 * max_width
    height = 3 * max_height

    if use_gpu:
        tiled_img = torch.zeros(
            (height, width, channel), dtype=torch.float32, device=device
        )
        filled_mask = torch.zeros((height, width), dtype=torch.uint8, device=device)
    else:
        tiled_img = np.zeros((height, width, channel), dtype=np.float32)
        filled_mask = np.zeros((height, width), dtype=np.uint8)

    for idx, cam_name in enumerate(cam_names):
        img = imgs[idx]

        if cam_name == "front_wide_110":
            tiled_img[max_height : 2 * max_height, 2 * max_width : 3 * max_width] = img
            filled_mask[max_height : 2 * max_height, 2 * max_width : 3 * max_width] = 1

        elif cam_name == "front_wide_60":
            tiled_img[:max_height, 2 * max_width : 3 * max_width] = img
            filled_mask[:max_height, 2 * max_width : 3 * max_width] = 1

        elif cam_name == "front_tele_30":
            tiled_img[:max_height, max_width : 2 * max_width] = img
            filled_mask[:max_height, max_width : 2 * max_width] = 1

        elif cam_name == "front_tele_15":
            tiled_img[:max_height, 3 * max_width : 4 * max_width] = img
            filled_mask[:max_height, 3 * max_width : 4 * max_width] = 1

        # elif cam_name == "front_wide_left_60":
        #     tiled_img[max_height : 2 * max_height, max_width : 2 * max_width] = img
        #     filled_mask[max_height : 2 * max_height, max_width : 2 * max_width] = 1

        elif cam_name == "front_left_99":
            tiled_img[max_height : 2 * max_height, :max_width] = img
            filled_mask[max_height : 2 * max_height, :max_width] = 1

        elif cam_name == "rear_left_99":
            tiled_img[2 * max_height :, max_width : 2 * max_width] = img
            filled_mask[2 * max_height :, max_width : 2 * max_width] = 1

        elif cam_name == "rear_left_30":
            tiled_img[
                2 * max_height : 2 * max_height + min_height,
                max_width - min_width : max_width,
            ] = img
            filled_mask[
                2 * max_height : 2 * max_height + min_height,
                max_width - min_width : max_width,
            ] = 1

        # elif cam_name == "front_wide_right_60":
        #     tiled_img[max_height : 2 * max_height, 3 * max_width : 4 * max_width] = img
        #     filled_mask[max_height : 2 * max_height, 3 * max_width : 4 * max_width] = 1

        elif cam_name == "front_right_99":
            tiled_img[max_height : 2 * max_height, 4 * max_width :] = img
            filled_mask[max_height : 2 * max_height, 4 * max_width :] = 1

        elif cam_name == "rear_right_99":
            tiled_img[
                2 * max_height : 3 * max_height, 3 * max_width : 4 * max_width
            ] = img
            filled_mask[
                2 * max_height : 3 * max_height, 3 * max_width : 4 * max_width
            ] = 1

        elif cam_name == "rear_right_30":
            tiled_img[
                2 * max_height : 2 * max_height + min_height,
                4 * max_width : 4 * max_width + min_width,
            ] = img
            filled_mask[
                2 * max_height : 2 * max_height + min_height,
                4 * max_width : 4 * max_width + min_width,
            ] = 1

        elif cam_name == "rear_50":
            tiled_img[
                2 * max_height : 3 * max_height, 2 * max_width : 3 * max_width
            ] = img
            filled_mask[
                2 * max_height : 3 * max_height, 2 * max_width : 3 * max_width
            ] = 1

    if use_gpu:
        min_y, max_y = (
            torch.where(filled_mask)[0].min(),
            torch.where(filled_mask)[0].max(),
        )
        min_x, max_x = (
            torch.where(filled_mask)[1].min(),
            torch.where(filled_mask)[1].max(),
        )
        tiled_img = tiled_img[min_y:max_y, min_x:max_x]

        tiled_img = tiled_img.clamp(0, 255)  # 确保值在 [0, 255] 范围内
        tiled_img = tiled_img.to(torch.uint8)  # 转换为 uint8
        tiled_img = tiled_img.cpu().numpy()  # 转换回 NumPy 数组
    else:
        # crop the image according to the lagrest filled area
        min_y, max_y = np.where(filled_mask)[0].min(), np.where(filled_mask)[0].max()
        min_x, max_x = np.where(filled_mask)[1].min(), np.where(filled_mask)[1].max()
        tiled_img = tiled_img[min_y:max_y, min_x:max_x]

        tiled_img = tiled_img.clip(0, 255)  # 确保值在 [0, 255] 范围内
        tiled_img = tiled_img.astype(np.uint8)

    return tiled_img

### Note(gls):not used
def layout_thoru(imgs: List[np.array], use_gpu=True) -> np.array:
    """Combine cameras into a tiled image.
    Layout:

        ##########################################################################################################
        #                   front_tele_30           # front_wide_60     # front_tele_15                          #
        #                   front_left_99           # front_wide_110    # front_right_99                         #
        # rear_left_30      # rear_left_99          # rear_50           # rear_right_99         # rear_right_30  #
        ##########################################################################################################
    """

    cam_names = [
        "front_wide_110",
        "front_wide_60",
        "front_tele_30",
        "front_tele_15",
        "front_left_99",
        "rear_left_99",
        "rear_left_30",
        "front_right_99",
        "rear_right_99",
        "rear_right_30",
        "rear_50",
    ]

    channel = imgs[0].shape[-1]
    if use_gpu:
        imgs = [torch.from_numpy(img).to(device, dtype=torch.float32) for img in imgs]

    max_width = 0
    max_height = 0
    min_width = 1e10
    min_height = 1e10

    for img in imgs:
        max_width = max(max_width, img.shape[1])
        max_height = max(max_height, img.shape[0])
        min_width = min(min_width, img.shape[1])
        min_height = min(min_height, img.shape[0])

    width = 5 * max_width
    height = 3 * max_height

    if use_gpu:
        tiled_img = torch.zeros(
            (height, width, channel), dtype=torch.float32, device=device
        )
        filled_mask = torch.zeros((height, width), dtype=torch.uint8, device=device)
    else:
        tiled_img = np.zeros((height, width, channel), dtype=np.float32)
        filled_mask = np.zeros((height, width), dtype=np.uint8)

    for idx, cam_name in enumerate(cam_names):
        img = imgs[idx]

        if cam_name == "front_wide_110":
            tiled_img[max_height : 2 * max_height, 2 * max_width : 3 * max_width] = img
            filled_mask[max_height : 2 * max_height, 2 * max_width : 3 * max_width] = 1

        elif cam_name == "front_wide_60":
            tiled_img[:max_height, 2 * max_width : 3 * max_width] = img
            filled_mask[:max_height, 2 * max_width : 3 * max_width] = 1

        elif cam_name == "front_tele_30":
            tiled_img[:max_height, max_width : 2 * max_width] = img
            filled_mask[:max_height, max_width : 2 * max_width] = 1

        elif cam_name == "front_tele_15":
            tiled_img[:max_height, 3 * max_width : 4 * max_width] = img
            filled_mask[:max_height, 3 * max_width : 4 * max_width] = 1

        elif cam_name == "front_left_99":
            tiled_img[max_height : 2 * max_height, max_width : 2 * max_width] = img
            filled_mask[max_height : 2 * max_height, max_width : 2 * max_width] = 1

        elif cam_name == "rear_left_99":
            tiled_img[2 * max_height :, max_width : 2 * max_width] = img
            filled_mask[2 * max_height :, max_width : 2 * max_width] = 1

        elif cam_name == "rear_left_30":
            tiled_img[
                2 * max_height : 2 * max_height + min_height,
                max_width - min_width : max_width,
            ] = img
            filled_mask[
                2 * max_height : 2 * max_height + min_height,
                max_width - min_width : max_width,
            ] = 1

        elif cam_name == "front_right_99":
            tiled_img[max_height : 2 * max_height, 3 * max_width : 4 * max_width] = img
            filled_mask[max_height : 2 * max_height, 3 * max_width : 4 * max_width] = 1

        elif cam_name == "rear_right_99":
            tiled_img[
                2 * max_height : 3 * max_height, 3 * max_width : 4 * max_width
            ] = img
            filled_mask[
                2 * max_height : 3 * max_height, 3 * max_width : 4 * max_width
            ] = 1

        elif cam_name == "rear_right_30":
            tiled_img[
                2 * max_height : 2 * max_height + min_height,
                4 * max_width : 4 * max_width + min_width,
            ] = img
            filled_mask[
                2 * max_height : 2 * max_height + min_height,
                4 * max_width : 4 * max_width + min_width,
            ] = 1

        elif cam_name == "rear_50":
            tiled_img[
                2 * max_height : 3 * max_height, 2 * max_width : 3 * max_width
            ] = img
            filled_mask[
                2 * max_height : 3 * max_height, 2 * max_width : 3 * max_width
            ] = 1

    if use_gpu:
        min_y, max_y = (
            torch.where(filled_mask)[0].min(),
            torch.where(filled_mask)[0].max(),
        )
        min_x, max_x = (
            torch.where(filled_mask)[1].min(),
            torch.where(filled_mask)[1].max(),
        )
        tiled_img = tiled_img[min_y:max_y, min_x:max_x]

        tiled_img = tiled_img.clamp(0, 255)  # 确保值在 [0, 255] 范围内
        tiled_img = tiled_img.to(torch.uint8)  # 转换为 uint8
        tiled_img = tiled_img.cpu().numpy()  # 转换回 NumPy 数组
    else:
        # crop the image according to the lagrest filled area
        min_y, max_y = np.where(filled_mask)[0].min(), np.where(filled_mask)[0].max()
        min_x, max_x = np.where(filled_mask)[1].min(), np.where(filled_mask)[1].max()
        tiled_img = tiled_img[min_y:max_y, min_x:max_x]

        tiled_img = tiled_img.clip(0, 255)  # 确保值在 [0, 255] 范围内
        tiled_img = tiled_img.astype(np.uint8)

    return tiled_img


def save_layout_video(video_dir, layout, use_gpu):
    pattern = os.path.join(video_dir, "*cam*.mp4")
    cam_paths: List[str] = glob.glob(pattern)

    cam_dict: Dict[int, str] = {}
    for path in cam_paths:
        basename = os.path.basename(path)
        # 提取 cam 后的数字，如 cam0.mp4 → 0,  front_cam5.mp4 → 5
        import re
        match = re.search(r'cam(\d+)', basename, re.IGNORECASE)
        if not match:
            continue
        cam_id = int(match.group(1))
        cam_dict[cam_id] = path

    # 排序确保 cam0, cam1, ... 顺序
    sorted_cam_ids = sorted(cam_dict.keys())

    frames_dict: Dict[int, List[np.ndarray]] = {}
    total_frames = None

    for cam_id in sorted_cam_ids:
        video_path = cam_dict[cam_id]
        imgs, _ = load_video(video_path)  # 你原有的 load_video 函数
        frames_dict[cam_id] = imgs

        if total_frames is None:
            total_frames = len(imgs)
        elif len(imgs) != total_frames:
            raise ValueError(
                f"相机 cam{cam_id} 帧数({len(imgs)}) 与 cam{sorted_cam_ids[0]} ({total_frames}) 不一致"
            )

    # 3. 按帧索引拼图写入
    video_save_path = os.path.join(video_dir, "layout.mp4")
    writer = imageio.get_writer(video_save_path, mode="I", fps=fps)

    for frame_idx in range(total_frames):
        # 按 cam_id 升序收集当前帧的图像
        frame_imgs = [
            frames_dict[cam_id][frame_idx]
            for cam_id in sorted_cam_ids
        ]
        tiled_img = layout(frame_imgs, use_gpu=use_gpu)
        writer.append_data(tiled_img)

    writer.close()


if __name__ == "__main__":
    video_dir = "output/qcraft_20250702_133223_Q2517/20251120_lidar+cam0_1_2_3_4_5_6_8_9_10_12/novel_traj/original_traj_step40000/videos"
    layout = layout_thoru

    save_layout_video(video_dir, layout, use_gpu=True)
