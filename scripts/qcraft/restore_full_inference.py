#!/usr/bin/env python3
"""
Fully restore original image resolution by pasting cropped inference back to original canvas.
Uses the original backup images to create complete, full-resolution results.
"""

import os
import numpy as np
from PIL import Image
import glob

def restore_cropped_inference(inference_img, backup_img, cam_id):
    """Restore a cropped inference image to full resolution using the backup."""

    # Convert to numpy arrays
    cropped = np.array(inference_img.convert('RGB'))
    original = np.array(backup_img.convert('RGB'))

    # Create result canvas (full size)
    result = original.copy()

    if cam_id == 4:
        # Camera 4: cropped 737x512 from 1024x512 (removed right 287 pixels)
        # Paste cropped region on left side, keep ego region untouched on right
        result[:, :cropped.shape[1]] = cropped

        # Add smooth transition at boundary (optional, remove if not needed)
        boundary_x = cropped.shape[1]  # 737
        transition_width = 30

        for x in range(boundary_x - transition_width, boundary_x):
            alpha = (x - (boundary_x - transition_width)) / transition_width
            result[:, x] = (result[:, x] * (1 - alpha) +
                           original[:, x] * alpha).astype(np.uint8)

    elif cam_id == 7:
        # Camera 7: cropped 731x512 from 1024x512 (removed left 293 pixels)
        # Paste cropped region on right side, keep ego region untouched on left
        cropped_width = cropped.shape[1]
        cropped_start_x = 1024 - cropped_width  # 1024 - 731 = 293

        result[:, cropped_start_x:] = cropped

        # Add smooth transition at boundary (optional, remove if not needed)
        boundary_x = cropped_start_x + cropped_width  # 1024
        transition_start = cropped_start_x - transition_width
        transition_width = 30

        for x in range(transition_start, cropped_start_x + transition_width):
            if x < cropped_start_x:
                alpha = 1.0
            else:
                alpha = (cropped_start_x + transition_width - x) / transition_width
                alpha = max(0, min(1, alpha))

            pos_in_cropped = x - cropped_start_x
            if 0 <= pos_in_cropped < cropped_width:
                result[:, x] = (result[:, x] * alpha +
                               cropped[:, pos_in_cropped] * (1 - alpha)).astype(np.uint8)

    return Image.fromarray(result)

def restore_cropped_video_frames(output_path, backup_dir, cam_id, frame_start=0, frame_end=-1):
    """Restore all cropped video frames for a camera."""

    # Find the video directory
    video_dir = os.path.join(output_path, "videos", f"trajectory_000_camera_{cam_id}")
    if not os.path.exists(video_dir):
        print(f"Error: Video directory not found: {video_dir}")
        return

    # Get all frames
    frame_files = sorted(glob.glob(os.path.join(video_dir, "frame_*.jpg")))

    if frame_end == -1:
        frame_end = len(frame_files)

    frame_files = frame_files[frame_start:frame_end]

    print(f"\nRestoring camera {cam_id}: {len(frame_files)} frames")
    print(f"From: {video_dir}")

    restored_dir = video_dir + "_full_resolution"
    os.makedirs(restored_dir, exist_ok=True)

    for i, frame_path in enumerate(frame_files):
        frame_name = os.path.basename(frame_path)

        # Find corresponding backup frame
        backup_frame_path = os.path.join(backup_dir, frame_name)

        if not os.path.exists(backup_frame_path):
            print(f"Warning: Backup not found for {frame_name}, skipping...")
            continue

        # Load images
        cropped_img = Image.open(frame_path)
        backup_img = Image.open(backup_frame_path)

        # Restore to full resolution
        restored_img = restore_cropped_inference(cropped_img, backup_img, cam_id)

        # Save
        restored_img.save(os.path.join(restored_dir, frame_name))

        print(f"  [{i+1}/{len(frame_files)}] {frame_name} -> {restored_img.size}")

    print(f"✓ Restored frames saved to: {restored_dir}")

def main():
    # Configuration
    CAMERAS = [4, 7]
    BACKUP_DIR = "data/qcraft/processed/training/20251025_163358_QCOYSD504206/images/_before_ego_mask_crop"
    OUTPUT_PATH = "output/qcraft_20251025_163358_QCOYSD504206/20251126_lidar+cam0_1_4_7"

    # Check backup directory
    if not os.path.exists(BACKUP_DIR):
        print(f"Error: Backup directory not found: {BACKUP_DIR}")
        print("You need to restore the original images from backup first!")
        return

    print("=" * 80)
    print("Cropped Inference Resolution Restorer")
    print("=" * 80)
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Backup dir:  {BACKUP_DIR}")
    print(f"Cameras:     {CAMERAS}")
    print("=" * 80)

    for cam_id in CAMERAS:
        restore_cropped_video_frames(OUTPUT_PATH, BACKUP_DIR, cam_id)

    print("\n" + "=" * 80)
    print("✓ All frames restored to full resolution!")
    print("Directories with '_full_resolution' contain complete images")
    print("=" * 80)

if __name__ == "__main__":
    main()