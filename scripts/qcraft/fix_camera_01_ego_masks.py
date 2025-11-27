#!/usr/bin/env python3
"""
Recrop ego masks for cameras 0 and 1 to correctly represent ego car body.
Based on actual ego mask analysis:
- Camera 0 mask is invalid (35% white pixels, no clear ego region)
- Camera 1 mask is invalid (8% white pixels, no clear ego region)
These cameras likely don't have ego car visible in the frame.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path

def create_empty_ego_mask(size=(512, 1024)):
    """Create an all-black ego mask (no ego car visible)."""
    mask = np.zeros(size, dtype=np.uint8)
    return Image.fromarray(mask)

def main():
    ego_dir = Path("data/qcraft/processed/training/20251025_163358_QCOYSD504206/ego_masks")

    print("=" * 80)
    print("Fixing Camera 0 and 1 Ego Masks")
    print("=" * 80)
    print(f"Ego mask directory: {ego_dir}")

    for cam_id in [0, 1]:
        mask_path = ego_dir / f"{cam_id}.png"
        if mask_path.exists():
            print(f"\nCamera {cam_id}:")
            print(f"  Existing mask: {mask_path}")

            # Save backup
            backup_path = ego_dir / f"{cam_id}_image.png"
            if not backup_path.exists():
                original = Image.open(mask_path)
                original.save(backup_path)
                print(f"  → Backup saved: {backup_path}")

            # Create empty black mask
            empty_mask = create_empty_ego_mask()
            empty_mask.save(mask_path)
            print(f"  ✓ Replaced with empty (black) mask - NO EGO CAR")
        else:
            print(f"\nCamera {cam_id}: No existing mask found")

    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)
    print("Cameras 0 and 1 ego masks have been set to empty (black).")
    print("This indicates no ego car is visible in these camera views.")
    print("\nNext steps:")
    print("1. Re-run preprocessing: qcraft_preprocess.py will skip cropping for cams 0,1")
    print("2. Camera 0 and 1 will keep their full 1024x512 resolution")
    print("3. Continue training as usual")
    print("=" * 80)

if __name__ == "__main__":
    main()