import os
import matplotlib.cm as cm
import imageio
import numpy as np
import cv2


def visualize_mask(image, mask, color):
    # 创建一个与输入图像相同大小的透明颜色层
    overlay = np.zeros_like(image, dtype=np.uint8)
    
    # 将掩码的非零区域设置为指定颜色（忽略透明度通道）
    mask_area = mask > 0
    overlay[mask_area] = color[:3]  # 只使用BGR通道
    
    # 计算Alpha混合
    alpha = 0.4  # 将透明度归一化到[0, 1]
    result = image.copy()
    
    # 对掩码区域进行Alpha混合
    for c in range(3):  # 对B, G, R三个通道分别处理
        result[mask_area, c] = (1 - alpha) * image[mask_area, c] + alpha * overlay[mask_area, c]
    
    return result


if __name__ == "__main__":
    data_path = "data/qcraft/processed/training/20251103_134932_QLC0N1000623_12337_12352"
    output_dir = "output/qcraft_20251103_134932_QLC0N1000623_12337_12352"

    num_cams = 13

    if os.path.exists(os.path.join(data_path, "lidar_pose")):
        total_frames = len(os.listdir(os.path.join(data_path, "lidar_pose")))

    save_dir = os.path.join(output_dir, "mask_vis")
    os.makedirs(save_dir, exist_ok=True)

    sky_color     = (135, 206, 235)
    road_color    = (128, 64, 128)
    human_color   = (220, 20, 60)
    vehicle_color = (0, 0, 142)

    for cam_id in range(num_cams):
        print(f"Visualizing masks for cam {cam_id}...")

        video_path = os.path.join(save_dir, f"cam_{cam_id}.mp4")
        writer = imageio.get_writer(video_path, mode="I", fps=10)

        for frame_id in range(total_frames):
            rgb_image_path = os.path.join(data_path, "images", f"{frame_id:06d}_{cam_id}.png")
            vis_image = cv2.imread(rgb_image_path)

            filename = f"{frame_id:06d}_{cam_id}.png"

            sky_mask_path = os.path.join(data_path, "sky_masks", filename)
            sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE)

            road_mask_path = os.path.join(data_path, "road_masks", filename)
            road_mask = cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE)

            human_mask_path = os.path.join(data_path, "dynamic_masks", "human", filename)
            human_mask = cv2.imread(human_mask_path, cv2.IMREAD_GRAYSCALE)

            vehicle_mask_path = os.path.join(data_path, "dynamic_masks", "vehicle", filename)
            vehicle_mask = cv2.imread(vehicle_mask_path, cv2.IMREAD_GRAYSCALE)

            vis_image = visualize_mask(vis_image, road_mask, road_color)
            vis_image = visualize_mask(vis_image, sky_mask, sky_color)
            vis_image = visualize_mask(vis_image, human_mask, human_color)
            vis_image = visualize_mask(vis_image, vehicle_mask, vehicle_color)

            # # save image
            # image_save_path = os.path.join(save_dir, f"{frame_id:03d}_{cam_id}.jpg")
            # cv2.imwrite(image_save_path, vis_image)

            # save video
            writer.append_data(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        writer.close()
        print(f"Saved visualization video for cam {cam_id} at {video_path}")