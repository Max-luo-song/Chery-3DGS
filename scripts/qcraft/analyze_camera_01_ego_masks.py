#!/usr/bin/env python3
"""
Analyze ego masks for cameras 0 and 1 to determine exact crop dimensions.

Usage:
    python scripts/qcraft/analyze_camera_01_ego_masks.py
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
import glob

def analyze_ego_mask(ego_mask_path):
    """
    Analyze ego mask to determine the crop strategy.

    Args:
        ego_mask_path: Path to ego mask image

    Returns:
        (crop_start_x, crop_end_x, ego_start_x, ego_end_x, ego_width)
    """
    # if not os.path.exists(ego_mask_path):
    #     print(f"Warning: ego_mask {ego_mask_path} not found")
    #     return None

    ego_mask = Image.open(ego_mask_path).convert('L')
    mask_array = np.array(ego_mask)

    # Find white region (ego car body) - pixels > 127
    white_mask = (mask_array > 127)
    cols_white = np.any(white_mask, axis=0)

    # Check if image is full white
    if np.all(cols_white):
        print(f"  ⚠️  WARNING: Entire image is white (ego everywhere) - likely no ego present")
        return None

    if not np.any(cols_white):
        print(f"  Warning: No white region found in {ego_mask_path}")
        return None

    ego_start = int(np.where(cols_white)[0][0])
    ego_end = int(np.where(cols_white)[0][-1])
    ego_width = ego_end - ego_start + 1
    ego_center = (ego_start + ego_end) // 2

    print(f"  Original image size: {ego_mask.size}")
    print(f"  Ego region: x=[{ego_start}, {ego_end}], width={ego_width}")

    # Determine which side has ego and crop the other side
    if ego_center < ego_mask.width // 2:  # White on LEFT
        # Ego on left, keep right part
        crop_start_x = ego_end + 1
        crop_end_x = ego_mask.width - 1
        print(f"  → Ego on LEFT, keeping RIGHT side: x=[{crop_start_x}, {crop_end_x}] (width={crop_end_x-crop_start_x+1})")
    else:  # White on RIGHT
        # Ego on right, keep left part
        crop_start_x = 0
        crop_end_x = ego_start - 1
        print(f"  → Ego on RIGHT, keeping LEFT side: x=[{crop_start_x}, {crop_end_x}] (width={crop_end_x-crop_start_x+1})")

    return (crop_start_x, crop_end_x, ego_start, ego_end, ego_width)

def main():
    print("=" * 80)
    print("Analyzing Ego Masks for Cameras 0 and 1")
    print("=" * 80)

    # Find data directory
    data_dirs = glob.glob('data/qcraft/processed/training/*/ego_masks')
    if not data_dirs:
        print("ERROR: No ego_masks directory found")
        print("Expected: data/qcraft/processed/training/*/ego_masks")
        return

    data_dir = data_dirs[0]
    ego_mask_dir = data_dir
    scene_dir = Path(data_dir).parent

    print(f"\nScene directory: {scene_dir}")
    print(f"Ego mask directory: {ego_mask_dir}")

    # Analyze cameras 0 and 1
    cameras = [0, 1]
    crop_results = {}

    print("\n" + "=" * 80)
    print("EGO MASK ANALYSIS RESULTS")
    print("=" * 80)

    for cam_id in cameras:
        print(f"\nCamera {cam_id}:")
        ego_mask_path = os.path.join(ego_mask_dir, f"{cam_id}.png")

        result = analyze_ego_mask(ego_mask_path)
        if result:
            crop_results[cam_id] = result

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for cam_id in cameras:
        if cam_id in crop_results:
            crop_start, crop_end, ego_start, ego_end, ego_width = crop_results[cam_id]
            cropped_width = crop_end - crop_start + 1
            crop_offset = crop_start

            print(f"\nCamera {cam_id}:")
            print(f"  → Crop offset: ({crop_offset}, 0)")
            print(f"  → Original size: (512, 1024)")
            print(f"  → Cropped size: (512, {cropped_width})")

            # Update logic for dataset_meta.py
            camera_name = "front_wide_110" if cam_id == 0 else "front_wide_60"
            print(f"\n  For dataset_meta.py:")
            print(f"    {cam_id}: {{")
            print(f'        "camera_name": "{camera_name}",')
            print(f'        "original_size": (512, {cropped_width}),')
            print(f'        "egocar_visible": False,')
            print(f'        "is_fisheye": False,')
            print(f"    }},")

            # Update logic for qcraft_preprocess.py
            print(f"\n  For qcraft_preprocess.py:")
            print(f"    {cam_id}: ({crop_offset}, 0),")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the crop dimensions above")
    print("2. Update datasets/dataset_meta.py with new original_size values")
    print("3. Verify that crop offsets in qcraft_preprocess.py are correct")
    print("4. Run crop_ego_mask_full.py to crop images and masks")
    print("=" * 80)

if __name__ == "__main__":
    main()