#!/usr/bin/env python3
"""
Script to crop OUT ego_mask regions (white areas) from original images for cameras 4 and 7.
This script will crop images to only keep the black regions (valid areas), removing white ego-car body.
The result is smaller resolution images focusing only on the valid content.

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
    """Verify that the ego_mask is valid and matches expected dimensions.
    Returns the ego_mask and the bounding box of the white region (to be blackened)."""
    if not os.path.exists(ego_mask_path):
        print(f"Warning: ego_mask not found at {ego_mask_path}")
        return None, (0, 0, 0, 0)

    ego_mask = Image.open(ego_mask_path).convert('L')
    mask_array = np.array(ego_mask)

    # Invert the mask (since we want to keep the black region, crop out the white region)
    # White pixels (255) become 1, black pixels (0) stay 0
    mask_array_inverted = (mask_array > 127).astype(mask_array.dtype)

    # Get bounding box of non-zero (white) region (region to crop OUT)
    rows = np.any(mask_array_inverted > 0, axis=1)
    cols = np.any(mask_array_inverted > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        print(f"Warning: ego_mask {ego_mask_path} has no valid white region (all black), skipping")
        return ego_mask, None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    bbox = (x_min, y_min, x_max, y_max)

    # Check dimensions
    if mask_array.shape != expected_shape:
        print(f"Warning: ego_mask shape {mask_array.shape} does not match image dimensions {expected_shape}")
        return ego_mask, None

    print(f"Verified ego_mask with white region to crop bbox={bbox}")
    return ego_mask, bbox

def crop_ego_from_image(image_path, ego_mask_path, crop_width_ratios=None, crop_height_ratios=None, verbose=False):
    """Crop OUT the ego_mask region (white area) from an image, keeping only black region (valid area).
    Returns a new cropped image with reduced resolution."""

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

    # Determine camera ID from path
    image_filename = os.path.basename(image_path)
    # Image name is typically something like "000_4.jpg"
    cam_id = int(image_filename.split('_')[-1].replace('.jpg', ''))

    # Crop image to keep only the valid region OUTSIDE the white ego_mask
    # Strategy: Since the white region is typically a contiguous block, we can:
    # 1. Calculate the inverted bounding box (i.e., the region we want to keep)
    # 2. For camera 4: keep left part (0 to x_min), for camera 7: keep right part (x_max to width)

    if cam_id == 7:  # Front right camera - ego car on right side
        # Keep the right side part (after the white region)
        cropped_image = image.crop((x_max, 0, image.width, image.height))
        x_offset = x_max
        y_offset = 0
    else:  # Camera 4 and others - ego car on left side
        # Keep the left side part (before the white region)
        cropped_image = image.crop((0, 0, x_min, image.height))
        x_offset = 0
        y_offset = 0

    print(f"Cropped image size: {cropped_image.size} (original: {image.size})")

    return cropped_image

def process_camera_sequence(data_dir, ego_mask_dir, camera_ids, backup_subdir="_before_ego_mask_crop"):
    """Process a camera sequence by cropping ego_mask regions (white areas) from images.
    White regions are removed, keeping only valid black regions with reduced resolution."""

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
    cameras_to_process = [0, 1,4, 7]

    print("=" * 80)
    print("Ego Mask Cropping Tool for Qcraft Dataset")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Ego mask directory: {ego_mask_dir}")
    print(f"Cameras to process: {cameras_to_process}")
    print("=" * 80)
    print("IMPORTANT: This script will:")
    print("1. Backup original images to 'images/_before_ego_mask_crop/'")
    print("2. Crop ego_mask white regions from images, keeping only valid black areas")
    print("3. Save cropped images in place with reduced resolution")
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