#!/usr/bin/env python3
"""
Render novel trajectory with ego_mask regions restored using alpha blending.
This creates natural-looking results by blending the original ego region with the trained renderings.
"""

import os
import numpy as np
from PIL import Image
import glob
import cv2

def load_ego_mask(ego_mask_path):
    """Load ego mask and return as alpha channel (white=opaque, black=transparent)."""
    if not os.path.exists(ego_mask_path):
        return None

    ego_mask = Image.open(ego_mask_path).convert('L')
    mask_array = np.array(ego_mask)

    # Invert: white in mask = ego car = should be masked out
    # Convert to alpha: 0 (black) = transparent, 255 (white) = opaque
    # We want ego region (white) to show original, so make it fully opaque
    alpha_mask = np.where(mask_array > 127, 255, 0).astype(np.uint8)

    return alpha_mask

def get_crop_coordinates_for_camera(cam_id, rendered_img):
    """Get the crop coordinates that were used for this camera."""
    original_width = 1024
    rendered_width = rendered_img.width

    if cam_id == 4:
        # Left camera - we cropped the right side
        # Original: 1024, Cropped: 737
        # We removed 1024-737=287 pixels from right side
        return {
            'left': 0,
            'top': 0,
            'right': original_width - 287,  # Keep left 737 pixels
            'bottom': 512
        }
    elif cam_id == 7:
        # Right camera - we cropped the left side
        # Original: 1024, Cropped: 731
        # We removed 1024-731=293 pixels from left side
        return {
            'left': 293,
            'top': 0,
            'right': original_width,
            'bottom': 512
        }
    else:
        return None

def create_blended_result(rendered_path, backup_path, ego_mask_path, cam_id, blend_ratio=0.3):
    """Create a blended result with ego region from backup and rendered scene."""

    # Load images
    rendered = Image.open(rendered_path).convert('RGB')
    original = Image.open(backup_path).convert('RGB')

    # Load ego mask (white=ego region, black=valid scene)
    ego_mask = load_ego_mask(ego_mask_path)

    if ego_mask is None:
        print(f"Warning: No ego mask found for camera {cam_id}")
        return rendered

    # Get crop coordinates
    crop_coords = get_crop_coordinates_for_camera(cam_id, rendered)

    if crop_coords is None:
        print(f"Warning: Unknown camera {cam_id}")
        return rendered

    # Create the full-resolution result
    result = np.array(original.copy())

    # Paste the rendered region back
    rendered_crop = np.array(rendered)

    # For camera 4: paste rendered on the left part
    if cam_id == 4:
        result[:, :rendered.width] = rendered_crop
        # Apply gradual alpha blending at the boundary
        blend_width = 50
        for x in range(rendered.width - blend_width, rendered.width):
            alpha = (x - (rendered.width - blend_width)) / blend_width
            result[:, x:x+1] = (result[:, x:x+1] * (1 - alpha) +
                               rendered_crop[:, x:x+1] * alpha).astype(np.uint8)

    # For camera 7: paste rendered on the right part
    elif cam_id == 7:
        x_start = original.width - rendered.width
        result[:, x_start:] = rendered_crop
        # Apply gradual alpha blending at the boundary
        blend_width = 50
        for x in range(x_start, x_start + blend_width):
            alpha = (x - x_start) / blend_width
            result[:, x:x+1] = (result[:, x:x+1] * alpha +
                               rendered_crop[:, x - x_start:x - x_start + 1] * (1 - alpha)).astype(np.uint8)

    # Apply ego mask to put back the ego car body
    # Dilate the mask a bit to ensure we include boundary pixels
    if ego_mask is not None:
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(ego_mask, kernel, iterations=2)

        # Create mask for blending: 0 (black) = use rendered, 1 (white) = use original ego region
        mask_float = dilated_mask.astype(float) / 255.0

        # Create inverted mask for smooth transition
        mask_inv = 1.0 - mask_float

        # Blend ego region softly
        ego_region_original = np.array(original)

        for c in range(3):
            result[:, :, c] = (result[:, :, c] * mask_inv +
                              ego_region_original[:, :, c] * mask_float)

    return Image.fromarray(result.astype(np.uint8))

def process_rendered_sequence(output_dir, backup_dir, ego_mask_dir, cameras=[4, 7]):
    """Process all rendered images to restore ego regions."""

    print("=" * 80)
    print("Restoring ego_mask regions in rendered images")
    print("=" * 80)

    output_path = Path(output_dir)

    for cam_id in cameras:
        print(f"\nProcessing camera {cam_id}...")

        ego_mask_path = os.path.join(ego_mask_dir, f"{cam_id}.png")
        backup_cam_dir = backup_dir
        rendered_cam_dir = output_path / f"render_000_camera_{cam_id}_00"

        if not rendered_cam_dir.exists():
            print(f"Warning: Rendered directory not found: {rendered_cam_dir}")
            continue

        # Get all rendered images
        rendered_files = sorted(glob.glob(str(rendered_cam_dir / "*.jpg")))

        if len(rendered_files) == 0:
            print(f"Warning: No rendered images found in {rendered_cam_dir}")
            continue

        # Create output directory for blended results
        output_blend_dir = output_path / f"render_000_camera_{cam_id}_00_with_ego"
        output_blend_dir.mkdir(exist_ok=True)

        print(f"Found {len(rendered_files)} rendered images")

        for i, rendered_path in enumerate(rendered_files):
            frame_name = os.path.basename(rendered_path)
            backup_path = os.path.join(backup_dir, frame_name)

            print(f"[{i+1}/{len(rendered_files)}] Blending {frame_name}...", end=" ")

            # Skip if backup doesn't exist
            if not os.path.exists(backup_path):
                print("Skipped (no backup)")
                continue

            try:
                blended = create_blended_result(rendered_path, backup_path,
                                              ego_mask_path, cam_id, blend_ratio=0.3)

                # Save blended result
                blended_path = output_blend_dir / frame_name
                blended.save(blended_path)

                print("Done")

            except Exception as e:
                print(f"Error: {e}")

    print("\n" + "=" * 80)
    print("Ego region restoration completed!")
    print(f"Blended results saved in: {output_dir}")
    print("Directories with '_with_ego' suffix contain the blended images")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Render novel trajectory with ego regions restored")
    parser.add_argument("--output_dir", type=str,
                       default="output/qcraft_20251025_163358_QCOYSD504206/20251126_lidar+cam0_1_4_7",
                       help="Output directory with rendered results")
    parser.add_argument("--backup_dir", type=str,
                       default="data/qcraft/processed/training/20251025_163358_QCOYSD504206/images/_before_ego_mask_crop",
                       help="Backup directory with original images")
    parser.add_argument("--ego_mask_dir", type=str,
                       default="data/qcraft/processed/training/20251025_163358_QCOYSD504206/ego_masks",
                       help="Directory with ego masks")
    parser.add_argument("--cameras", type=int, nargs='+', default=[4, 7],
                       help="Camera IDs to process")

    args = parser.parse_args()

    # Check if backup directory exists
    if not os.path.exists(args.backup_dir):
        print(f"Error: Backup directory not found: {args.backup_dir}")
        print("You need to restore the original images first!")
        exit(1)

    process_rendered_sequence(args.output_dir, args.backup_dir,
                             args.ego_mask_dir, args.cameras)