import os
import matplotlib.cm as cm
import imageio
import numpy as np
import cv2


def visualize_dynamic_masks(image, dynamic_mask, color):
    # 创建一个与输入图像相同大小的透明颜色层
    overlay = np.zeros_like(image, dtype=np.uint8)
    
    # 将掩码的非零区域设置为指定颜色（忽略透明度通道）
    mask_area = dynamic_mask > 0
    overlay[mask_area] = color[:3]  # 只使用BGR通道
    
    # 计算Alpha混合
    alpha = 0.5  # 将透明度归一化到[0, 1]
    result = image.copy()
    
    # 对掩码区域进行Alpha混合
    for c in range(3):  # 对B, G, R三个通道分别处理
        result[mask_area, c] = (1 - alpha) * image[mask_area, c] + alpha * overlay[mask_area, c]
    
    return result


if __name__ == "__main__":
    data_path = "data/qcraft/processed/training/20250902_163634_Q3703"
    output_dir = "output/qcraft_20250902_163634_Q3703"

    num_cams = 13
    # data_path = "data/chery/processed/training/clip_1746752396800"
    # output_dir = "output/chery_clip_1746752396800"

    # num_cams = 7

    if os.path.exists(os.path.join(data_path, "lidar_pose")):
        total_frames = len(os.listdir(os.path.join(data_path, "lidar_pose")))

    save_dir = os.path.join(output_dir, "mask_vis")
    os.makedirs(save_dir, exist_ok=True)

    human_color = (0, 165, 255)  # 亮橙色
    vehicle_color = (255, 150, 0)  # 蓝色

    for cam_id in range(num_cams):
        print(f"Visualizing dynamic mask for cam {cam_id}...")

        video_path = os.path.join(save_dir, f"cam_{cam_id}.mp4")
        writer = imageio.get_writer(video_path, mode="I", fps=10)

        for frame_id in range(total_frames):
            rgb_image_path = os.path.join(
                data_path, "images", f"{frame_id:03d}_{cam_id}.jpg"
            )
            rgb = cv2.imread(rgb_image_path)

            human_mask_path = os.path.join(
                data_path, "dynamic_masks", "human", f"{frame_id:03d}_{cam_id}.png"
            )
            human_mask = cv2.imread(human_mask_path, cv2.IMREAD_GRAYSCALE)

            vehicle_mask_path = os.path.join(
                data_path, "dynamic_masks", "vehicle", f"{frame_id:03d}_{cam_id}.png"
            )
            vehicle_mask = cv2.imread(vehicle_mask_path, cv2.IMREAD_GRAYSCALE)

            vis_image = visualize_dynamic_masks(
                rgb, human_mask, human_color
            )
            vis_image = visualize_dynamic_masks(
                vis_image, vehicle_mask, vehicle_color
            )

            # # save image
            # image_save_path = os.path.join(save_dir, f"{frame_id:03d}_{cam_id}.jpg")
            # cv2.imwrite(image_save_path, vis_image)

            # save video
            writer.append_data(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        writer.close()
        print(f"Saved lidar visualization video for cam {cam_id} at {video_path}")
