#!/usr/bin/env python3
"""
Camera calibration adjustment tool for ego-mask cropping.

This script adjusts the camera intrinsic matrix (K matrix) to account for
the image cropping we did to remove ego car body regions.

For cameras where we cropped the image, we need to:
1. Adjust the principal point (cx, cy) to the new image center
2. Keep focal lengths (fx, fy) unchanged (since we only cropped, no scaling)

Usage:
    python scripts/qcraft/update_camera_calibration.py
"""

import os
import json
import numpy as np
from pathlib import Path

def load_camera_parameters(param_path):
    """Load camera parameters from json file."""
    with open(param_path, 'r') as f:
        return json.load(f)

def save_camera_parameters(param_path, params):
    """Save camera parameters to json file."""
    with open(param_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Updated: {param_path}")

def adjust_intrinsics_for_crop(full_width, full_height, cropped_width, cropped_height,
                                original_params):
    """
    Adjust camera intrinsics when image is cropped (no scaling, just crop).

    Cropping changes:
      - cx: shift horizontally based on crop offset
      - cy: shift vertically based on crop offset (but we didn't crop vertically)

    Camera 4: Remove right side, keep left → cx shift to the left
    Camera 7: Remove left side, keep right → cx shift to the left

    Args:
        full_width, full_height: Original image dimensions
        cropped_width, cropped_height: Cropped image dimensions
        original_params: Original camera parameter dict

    Returns:
        adjusted_params: Camera parameters adjusted for cropping
        crop_info: Info about the crop
    """

    # Extract original intrinsic matrix
    K_original = np.array(original_params['K'])

    # Original principal points
    cx_orig = K_original[0, 2]
    cy_orig = K_original[1, 2]
    fx_orig = K_original[0, 0]
    fy_orig = K_original[1, 1]

    # Calculate crop parameters based on camera
    if cropped_width == 737:  # Camera 4
        # Crop: keep left 737 pixels, remove right 287 pixels
        crop_offset_x = 0  # Start from x=0
        crop_offset_y = 0
        width_delta = full_width - cropped_width  # 287 pixels removed from right
        height_delta = 0
    elif cropped_width == 731:  # Camera 7
        # Crop: remove left 293 pixels, keep right 731 pixels
        crop_offset_x = full_width - cropped_width  # 293 pixels removed from left
        crop_offset_y = 0
        width_delta = 293
        height_delta = 0
    else:
        # No crop
        return original_params, None

    # Calculate new principal point
    # New cx = original cx - crop_offset_x
    cx_new = cx_orig - crop_offset_x
    cy_new = cy_orig - crop_offset_y  # Should stay the same

    # Focal lengths remain unchanged (no scaling, just crop)
    fx_new = fx_orig
    fy_new = fy_orig

    # Build new intrinsic matrix
    K_new = np.array([
        [fx_new, 0.0,    cx_new],
        [0.0,    fy_new, cy_new],
        [0.0,    0.0,    1.0]
    ])

    # Update parameters
    adjusted_params = original_params.copy()
    adjusted_params['K'] = K_new.tolist()

    # Store crop info for verification
    crop_info = {
        'full_size': (full_width, full_height),
        'cropped_size': (cropped_width, cropped_height),
        'crop_offset': (crop_offset_x, crop_offset_y),
        'cx_shift': cx_new - cx_orig,
        'cy_shift': cy_new - cy_orig,
        'original_fx': fx_orig,
        'original_fy': fy_orig,
        'new_fx': fx_new,
        'new_fy': fy_new
    }

    return adjusted_params, crop_info

def process_camera_calibration(data_dir, cameras=[4, 7], dry_run=False):
    """Process and update camera calibration files."""

    print("=" * 80)
    print("Camera Calibration Adjustment Tool")
    print("=" * 80)
    print(f"Data dir: {data_dir}")
    print(f"Cameras to adjust: {cameras}")
    print(f"Dry run: {dry_run}")
    print("=" * 80)

    intrinsics_dir = Path(data_dir) / "intrinsics"

    if not intrinsics_dir.exists():
        print(f"Error: Intrinsics directory not found at {intrinsics_dir}")
        return

    crop_info_all = {}

    # Full size from camera meta
    # Camera 4 and 7 have original size of 1024x512
    original_width, original_height = 1024, 512

    cropped_sizes = {
        4: 737,  # Camera 4 cropped to 737x512
        7: 731   # Camera 7 cropped to 731x512
    }

    # Read intrinsics and store original values
    original_values = {}

    for cam_id in cameras:
        intrinsic_file = intrinsics_dir / f"{cam_id}.txt"

        if not intrinsic_file.exists():
            print(f"Warning: Camera {cam_id} intrinsics not found at {intrinsic_file}")
            continue

        print(f"\nCamera {cam_id}:")
        print(f"  Original size: {full_width}x{full_height}")
        print(f"  Cropped size:  {cropped_width}x{cropped_height}")

        # Adjust intrinsics
        adjusted_params, crop_info = adjust_intrinsics_for_crop(
            full_width, full_height,
            cropped_width, cropped_height,
            cam_params
        )

        if crop_info is not None:
            # Save adjusted parameters
            all_params[str(cam_id)] = adjusted_params
            crop_info_all[str(cam_id)] = crop_info

            # Print adjustment info
            print(f"  Principal point (cx, cy): ({cam_params['K'][0][2]:.2f}, {cam_params['K'][1][2]:.2f})")
            print(f"  ↓ Adjusted to:")
            print(f"  Principal point (cx, cy): ({adjusted_params['K'][0][2]:.2f}, {adjusted_params['K'][1][2]:.2f})")
            print(f"  Shift: ({crop_info['cx_shift']:.2f}, {crop_info['cy_shift']:.2f})")
            print(f"  Focal lengths unchanged: ({crop_info['fx_orig']:.2f}, {crop_info['fy_orig']:.2f})")
        else:
            print(f"  No adjustment needed (no crop)")

    # Save updated parameters
    if not dry_run:
        save_camera_parameters(param_file_path, all_params)

    print("\n" + "=" * 80)
    print("Camera calibration adjustment completed!")
    print(f"Updated file: {param_file_path}")
    print("=" * 80)

    # Save crop info
    crop_info_path = param_file_path.parent / "crop_info.json"
    with open(crop_info_path, 'w') as f:
        json.dump(crop_info_all, f, indent=2)
    print(f"Crop info saved: {crop_info_path}")

    return crop_info_all

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adjust camera calibration after cropping")
    parser.add_argument("--data_dir", type=str,
                       default="data/qcraft/processed/training/20251025_163358_QCOYSD504206",
                       help="Data directory containing cam_params_undistort.json")
    parser.add_argument("--cameras", type=int, nargs='+', default=[4, 7],
                       help="Camera IDs to adjust")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't save changes, just print what would be done")

    args = parser.parse_args()

    crop_info = process_camera_calibration(args.data_dir, args.cameras, args.dry_run)

    if args.dry_run:
        print("\n" + "=" * 80)
        print("NOTE: This was a dry run. No files were modified.")
        print("Remove --dry-run flag to apply changes.")
        print("=" * 80)