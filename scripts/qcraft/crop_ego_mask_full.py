#!/usr/bin/env python3
"""
Comprehensive ego mask cropping tool for cameras 0, 1, 4, and 7.

This script crops based on predefined camera configurations:
- Camera 0 (front_wide_110): crop bottom part
- Camera 1 (front_wide_60): crop bottom part  
- Camera 4 (front_left_99): crop right part
- Camera 7 (front_right_99): crop left part

All files are backed up before cropping.

Usage:
    python scripts/qcraft/crop_ego_mask_full.py

After cropping, use restore_full_inference.py to restore original resolutions for inference.
"""

import os
import numpy as np
from PIL import Image
import glob
from pathlib import Path
import shutil

# Camera configurations with crop strategies
CAMERA_CONFIGS = {
    0: {
        "camera_name": "front_wide_110",
        "original_size": (342, 1024),
        "crop_strategy": "bottom",
        "egocar_visible": False,
        "is_fisheye": False,
    },
    1: {
        "camera_name": "front_wide_60",
        "original_size": (472, 1024),
        "crop_strategy": "bottom",
        "egocar_visible": False,
        "is_fisheye": False,
    },
    4: {
        "camera_name": "front_left_99",
        # "original_size": (512, 772),
        "original_size": (512, 200),
        "crop_strategy": "right",
        "egocar_visible": False,
        "is_fisheye": False,
    },
    7: {
        "camera_name": "front_right_99",
        # "original_size": (512, 765),1024
        "original_size": (512, 200),
        "crop_strategy": "left",
        "egocar_visible": False,
        "is_fisheye": False,
    },
}

# ==============================================================================
# CROP COORDINATES CALCULATION
# ==============================================================================

def calculate_crop_coords(camera_id, image_size):
    """
    Calculate crop coordinates based on camera configuration.
    
    注意：config中的original_size格式为 (target_height, target_width)
    PIL的image_size格式为 (width, height)
    """
    config = CAMERA_CONFIGS.get(camera_id)
    if not config:
        return None

    # 从配置中获取目标尺寸 (高, 宽)
    target_height, target_width = config["original_size"]
    strategy = config["crop_strategy"]

    # 从PIL获取实际尺寸 (宽, 高)
    current_width, current_height = image_size

    if strategy == "bottom":
        # 从顶部开始，向下裁剪到 target_height 的高度
        return (0, 0, current_width, target_height)

    elif strategy == "right":
        # 从左侧开始，向右裁剪到 target_width 的宽度
        return (0, 0, target_width, current_height)

    elif strategy == "left":
        # 从右侧开始，向左裁剪到 target_width 的宽度
        start_x = current_width - target_width
        return (start_x, 0, current_width, current_height)

    return None

# ==============================================================================
# CROPPING FUNCTIONS
# ==============================================================================

def crop_image(image_path, crop_coords):
    """
    Crop image based on crop coordinates.

    Args:
        image_path: Path to image
        crop_coords: (left, top, right, bottom) for PIL crop

    Returns:
        Cropped PIL Image
    """
    img = Image.open(image_path)
    cropped_img = img.crop(crop_coords)
    return cropped_img

def crop_mask(mask_path, crop_coords):
    """
    Crop mask (dynamic mask, sky mask) based on crop coordinates.

    Args:
        mask_path: Path to mask
        crop_coords: (left, top, right, bottom) for PIL crop

    Returns:
        Cropped PIL Image
    """
    mask = Image.open(mask_path)
    cropped_mask = mask.crop(crop_coords)
    return cropped_mask

# ==============================================================================
# BACKUP AND PROCESS
# ==============================================================================

def backup_and_crop_images(data_dir, cameras):
    """Backup and crop images for specified cameras."""

    images_dir = Path(data_dir) / "images"
    backup_dir = Path(data_dir) / "images" / "_before_ego_mask_crop"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STEP 1: Processing IMAGES")
    print("=" * 80)

    for cam_id in cameras:
        config = CAMERA_CONFIGS.get(cam_id)
        if not config:
            print(f"\nCamera {cam_id}: No configuration found, skipping")
            continue

        print(f"\nCamera {cam_id} ({config['camera_name']}):")
        print(f"  Original size: {config['original_size']}")
        print(f"  Crop strategy: {config['crop_strategy']}")

        # Get all images for this camera
        camera_images_pattern = images_dir / f"*_{cam_id}.jpg"
        image_files = sorted(glob.glob(str(camera_images_pattern)))

        if not image_files:
            print(f"  → No images found for camera {cam_id}")
            continue

        processed_count = 0

        for i, image_path in enumerate(image_files, 1):
            image_filename = Path(image_path).name
            backup_path = backup_dir / image_filename

            print(f"  [{i}/{len(image_files)}] Processing {image_filename}...", end="")

            # Backup or load from backup
            if backup_path.exists():
                original_img = Image.open(backup_path)
                print("Recropping from backup")
            else:
                original_img = Image.open(image_path)
                original_img.save(backup_path)
                print("Backup + crop")

            # Calculate crop coordinates based on actual image size
            crop_coords = calculate_crop_coords(cam_id, original_img.size)
            if crop_coords:
                cropped_img = original_img.crop(crop_coords)
                cropped_img.save(image_path)  # Overwrite current image
                processed_count += 1

        print(f"  → Processed {processed_count} images for Camera {cam_id}")

    print("✓ Image processing complete")

def backup_and_crop_masks(data_dir, cameras):
    """Backup and crop dynamic masks for all specified cameras."""

    backup_dir = Path(data_dir) / "dynamic_masks" / "_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STEP 2: Processing DYNAMIC MASKS")
    print("=" * 80)

    # Process dynamic masks: all, human, vehicle, car, rider, dynamic_objects
    mask_types = ["all", "human", "vehicle", "car", "rider", "dynamic_objects"]

    for cam_id in cameras:
        config = CAMERA_CONFIGS.get(cam_id)
        if not config:
            continue

        print(f"\nCamera {cam_id} ({config['camera_name']}):")

        for mask_type in mask_types:
            mask_dir = Path(data_dir) / "dynamic_masks" / mask_type

            if not mask_dir.exists():
                print(f"  {mask_type}: Directory not found, skipping")
                continue

            # Pattern: *_cam*.jpg
            mask_pattern = mask_dir / f"*_{cam_id}.png"
            mask_files = sorted(glob.glob(str(mask_pattern)))

            if not mask_files:
                print(f"  {mask_type}: No files found")
                continue

            mask_backup_dir = backup_dir / mask_type
            mask_backup_dir.mkdir(parents=True, exist_ok=True)

            processed_count = 0
            for mask_path in mask_files:
                mask_filename = Path(mask_path).name
                mask_backup_path = mask_backup_dir / mask_filename

                # Backup or crop from backup
                if mask_backup_path.exists():
                    mask_img = Image.open(mask_backup_path)
                else:
                    mask_img = Image.open(mask_path)
                    mask_img.save(mask_backup_path)

                # Calculate crop coordinates
                crop_coords = calculate_crop_coords(cam_id, mask_img.size)
                if crop_coords:
                    cropped_mask = mask_img.crop(crop_coords)
                    cropped_mask.save(mask_path)
                    processed_count += 1

            if processed_count > 0:
                print(f"  {mask_type}: Processed {processed_count} masks")

    print("✓ Dynamic mask processing complete")

def backup_and_crop_sky_masks(data_dir, cameras):
    """Backup and crop sky masks for all specified cameras."""

    print("\n" + "=" * 80)
    print("STEP 3: Processing SKY MASKS")
    print("=" * 80)

    sky_dir = Path(data_dir) / "sky_masks"
    backup_dir = Path(data_dir) / "sky_masks" / "_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not sky_dir.exists():
        print("Skipping sky masks: Directory not found")
        return

    for cam_id in cameras:
        config = CAMERA_CONFIGS.get(cam_id)
        if not config:
            continue

        print(f"\nCamera {cam_id} ({config['camera_name']}):")

        # Pattern: *_cam*.jpg
        sky_mask_pattern = sky_dir / f"*_{cam_id}.png"
        sky_files = sorted(glob.glob(str(sky_mask_pattern)))

        if not sky_files:
            print(f"  → No sky mask files found")
            continue

        processed_count = 0
        for sky_path in sky_files:
            sky_filename = Path(sky_path).name
            sky_backup_path = backup_dir / sky_filename

            # Backup or crop from backup
            if sky_backup_path.exists():
                sky_img = Image.open(sky_backup_path)
            else:
                sky_img = Image.open(sky_path)
                sky_img.save(sky_backup_path)

            # Calculate crop coordinates
            crop_coords = calculate_crop_coords(cam_id, sky_img.size)
            if crop_coords:
                cropped_sky = sky_img.crop(crop_coords)
                cropped_sky.save(sky_path)
                processed_count += 1

        if processed_count > 0:
            print(f"  → Processed {processed_count} sky masks")

    print("✓ Sky mask processing complete")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    # Configuration
    data_dir = "data/qcraft/processed/training/20251025_163358_QCOYSD504206"
    cameras = [0, 1, 4, 7]

    print("=" * 80)
    print("Comprehensive Ego Mask Cropping Tool")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Cameras to process: {cameras}")
    
    # Print camera configurations
    print("\nCamera configurations:")
    for cam_id in cameras:
        config = CAMERA_CONFIGS.get(cam_id)
        if config:
            print(f"  Camera {cam_id} ({config['camera_name']}):")
            print(f"    Original size: {config['original_size']}")
            print(f"    Crop strategy: {config['crop_strategy']}")
    
    print("=" * 80)
    print("This script will:")
    print("1. Back up ALL original images, masks, and dynamic masks")
    print("2. Crop images based on camera configuration")
    print("3. Crop dynamic masks (all, human, vehicle, car, rider, dynamic_objects)")
    print("4. Crop sky masks (if present)")
    print("WARNING: This will permanently crop files to smaller resolution")
    print("Backup directories:")
    print("  - images/_before_ego_mask_crop/")
    print("  - dynamic_masks/_backup/")
    print("  - sky_masks/_backup/")
    print("=" * 80)

    confirm = input("\nDo you want to continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Aborted by user.")
        return

    # Step 1: Backup and crop images
    backup_and_crop_images(data_dir, cameras)

    # Step 2: Backup and crop dynamic masks
    backup_and_crop_masks(data_dir, cameras)

    # Step 3: Backup and crop sky masks
    backup_and_crop_sky_masks(data_dir, cameras)

    print("\n" + "=" * 80)
    print("ALL PROCESSING COMPLETED!")
    print("=" * 80)
    print("Files cropped:")
    print(f"  - Images: {cameras}")
    print(f"  - Dynamic masks: all, human, vehicle, car, rider, dynamic_objects")
    print(f"  - Sky masks (if present)")
    print()
    print("Backups saved at:")
    print(f"  - {data_dir}/images/_before_ego_mask_crop/")
    print(f"  - {data_dir}/dynamic_masks/_backup/")
    print(f"  - {data_dir}/sky_masks/_backup/")
    print()
    print("Next steps:")
    print("1. Re-run qcraft_preprocess to update intrinsics")
    print("2. Train model")
    print("3. After rendering, use restore_full_inference.py to restore original resolution")
    print("=" * 80)

if __name__ == "__main__":
    main()
