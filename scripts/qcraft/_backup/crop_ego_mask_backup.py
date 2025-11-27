
#!/usr/bin/env python3
"""
Script to crop ego_mask regions from original images for cameras 4 and 7.
This script will extract and save the ego_mask portion from images to remove ego-car body from the training data.

Usage:
    python scripts/qcraft/crop_ego_mask.py

This creates backup copies of processed files.
"""

import os
import numpy as np
from PIL import Image
import glob
from pathlib import Path

def verify_ego_mask(ego_mask_path, expected_shape):
    """Verify that the ego_mask is valid and matches expected dimensions."""
    if not os.path.exists(ego_mask_path):
        print(f"Warning: ego_mask not found at {ego_mask_path}")
        return None, (0, 0, 0, 0)

    ego_mask = Image.open(ego_mask_path).convert('L')
    mask_array = np.array(ego_mask)

    # Check if the ego_mask is essentially blank (all zeros or very few pixels)
    non_zeros_pixel_count = np.sum(mask_array > 0)
    total_pixels = mask_array.size
    non_zeros_ratio = non_zeros_pixel_count / total_pixels

    print(f"For ego_mask in {ego_mask_path}: {total_pixels} total pixels, {non_zeros_pixel_count} non-zero pixels, {100*non_zeros_ratio:.2f}%")

    if non_zeros_pixel_count < 1000:
        print(f"Warning: ego_mask {ego_mask_path} has very few non-zero pixels ({non_zeros_pixel_count} < 1000), skipping")
        return ego_mask, None

    # Get bounding box of non-zero region
    rows = np.any(mask_array > 0, axis=1)
    cols = np.any(mask_array > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        print(f"Warning: ego_mask {ego_mask_path} has no valid mask region, skipping")
        return ego_mask, None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    bbox = (x_min, y_min, x_max, y_max)

    # Check dimensions
    if mask_array.shape != expected_shape:
        print(f"Warning: ego_mask shape {mask_array.shape} does not match image dimensions {expected_shape}")
        return ego_mask, None

    print(f"Verified ego_mask with valid region bbox={bbox}")
    return ego_mask, bbox

def crop_ego_from_image(image_path, ego_mask_path, crop_width_ratios=None, crop_height_ratios=None, verbose=False):
    """Crop the ego_mask region from an image and save to a new subfolder."""

    # Load image
    if not os.path.exists(image_path):
        print(f"Error: image not found at {image_path}")
        return None

    image = Image.open(image_path)
    image_array = np.array(image)
    original_shape = image_array.shape[:2]
    expected_shape = original_shape if len(image_array.shape) == 2 else original_shape

    # Verify ego_mask
    ego_mask, bbox = verify_ego_mask(ego_mask_path, expected_shape)

    if bbox is None:
        # Skip images with invalid mask, but make a backup if needed
        return None

    x_min, y_min, x_max, y_max = bbox

    # Expand bbox to include more surrounding area
    if crop_width_ratios is not None:
        left_add = int(crop_width_ratios[0] * (x_max - x_min))
        right_add = int(crop_width_ratios[1] * (x_max - x_min))
        x_min = max(0, x_min - left_add)
        x_max = min(image.width, x_max + right_add)

    if crop_height_ratios is not None:
        top_add = int(crop_height_ratios[0] * (y_max - y_min))
        bottom_add = int(crop_height_ratios[1] * (y_max - y_min))
        y_min = max(0, y_min - top_add)
        y_max = min(image.height, y_max + bottom_add)

    # Add safety headroom
    safety_pixels_x = 10
    safety_pixels_y = 5
    x_min = max(0, x_min - safety_pixels_x)
    x_max = min(image.width, x_max + safety_pixels_x)
    y_min = max(0, y_min - safety_pixels_y)
    y_max = min(image.height, y_max + safety_pixels_y)

    if verbose:
        # Show extracted region info
        print(f"Image size: {image.size}")
        print(f"Mask bbox (raw): {bbox}")
        print(f"Mask bbox (expanded): x=[{x_min},{x_max}], y=[{y_min},{y_max}]")
        print(f"Ego region extent: {x_max-x_min} pixels horizontally, {y_max-y_min} pixels vertically")

    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    mask_crop = ego_mask.crop((x_min, y_min, x_max, y_max))
    background = Image.new("RGB", image.size, (0, 0, 0))
    background.paste(cropped_image, (x_min, y_min))

    return background

def process_camera_sequence(data_dir, ego_mask_dir, camera_ids, backup_subdir="_before_ego_mask_crop"):
    """Process a camera sequence by cropping ego_mask regions from images."""

    data_path = Path(data_dir)
    images_dir = data_path / "images"
    backup_dir = data_path / "images" / backup_subdir

    # Create backup directory if it doesn't exist
    backup_dir.mkdir(parents=True, exist_ok=True)

    for cam_id in camera_ids:
        # Get fixed ego_mask for this camera
        ego_mask_path = os.path.join(ego_mask_dir, f"{cam_id}.png")

        if not os.path.exists(ego_mask_path):
            print(f"Warning: ego_mask {ego_mask_path} not found, skipping camera {cam_id}")
            continue

        print(f"\nProcessing camera {cam_id}...")

        # Get all images for this camera
        camera_images_pattern = os.path.join(images_dir, f"*_{cam_id}.jpg")
        image_files = sorted(glob.glob(camera_images_pattern))

        total_images = len(image_files)
        if total_images == 0:
            print(f"Warning: No images found for camera {cam_id}")
            continue

        print(f"Found {total_images} images for camera {cam_id}")

        for i, image_path in enumerate(image_files, 1):
            image_filename = os.path.basename(image_path)
            frame_idx = image_filename.replace(f"_{cam_id}.jpg", "")

            # Backup original if not already backed up
            backup_path = backup_dir / image_filename
            if not backup_path.exists():
                os.rename(image_path, backup_path)

                print(f"[{i}/{total_images}] Processing {image_filename}...", end="")

                # Extract region and save
                cropped_image = crop_ego_from_image(str(backup_path), ego_mask_path, crop_width_ratios=[0.1, 0.1], crop_height_ratios=[0.05, 0.2])

                if cropped_image is not None:
                    cropped_image.save(image_path)
                    print("Done")
                else:
                    # If no valid mask or extraction failed, restore original
                    os.rename(backup_path, image_path)
                    print("Skipped (no valid ego_mask)")
            else:
                print(f"[{i}/{total_images}] Skipping {image_filename} (already processed)")

def main():
    """Main function to process the Qcraft dataset."""

    # Configuration - UPDATE THESE PATHS IF NEEDED
    data_dir = "data/qcraft/processed/training/20251025_163358_QCOYSD504206"
    ego_mask_dir = os.path.join(data_dir, "ego_masks")

    # Process only cameras 4 and 7 as requested
    cameras_to_process = [4, 7]

    print("=" * 80)
    print("Ego Mask Cropping Tool for Qcraft Dataset")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Ego mask directory: {ego_mask_dir}")
    print(f"Cameras to process: {cameras_to_process}")
    print("=" * 80)
    print("IMPORTANT: This script will:")
    print("1. Backup original images to 'images/_before_ego_mask_crop/'")
    print("2. Crop ego_mask regions from images")
    print("3. Save cropped images in place")
    print("=" * 80)

    confirm = input("\nDo you want to continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Aborted by user.")
        return

    # Process all images
    process_camera_sequence(data_dir, ego_mask_dir, cameras_to_process)

    print("\n" + "=" * 80)
    print("Ego mask cropping completed!")
    print(f"Original files backed up to: {data_dir}/images/_before_ego_mask_crop/")
    print("=" * 80)

if __name__ == "__main__":
    main()