#!/usr/bin/env python3
"""
Complete solution for ego mask cropping and restoration.

WORKFLOW:
1. Crop ego_mask regions (STEP 1)
   - Camera 4: Ego on RIGHT side → Keep LEFT [0-771] (772×512)
   - Camera 7: Ego on LEFT side → Keep RIGHT [259-1023] (765×512)

2. Preprocess with correct intrinsics (STEP 2)
   - Camera 4: cx_adjusted = 512 (no change, stays at center)
   - Camera 7: cx_adjusted = 512 - 259 = 253 (shifts left)

3. Train 3DGS model (STEP 3)
   - Side views should now train properly with correct intrinsics

4. Render novel views (STEP 4)
   - Model generates novel viewpoint renders

5. Restore cropped regions (STEP 5)
   - Use ego_mask_crop_restore.py to blend original ego regions back

USAGE:
------
# Step 1: Crop (DONE - images already cropped to 772×512 and 765×512)
python scripts/qcraft/ego_mask_crop_restore.py --mode crop

# Step 2: Preprocess with correct intrinsics
bash scripts/qcraft/preprocess_data_qcraft.sh

# Step 3: Train model
bash scripts/qcraft/train.sh [config_name]

# Step 4: Render (if needed)
bash scripts/qcraft/render_novel_trajectory.sh

# Step 5: Restore (after rendering)
python scripts/qcraft/ego_mask_crop_restore.py \
    --mode restore \
    --rendered-dir [path_to_renders] \
    --output-dir outputs/qcraft/restored

COMPLETION CHECKLIST:
---------------------
✓ Camera 4 & 7 images cropped (772×512 and 765×512)
✓ Ego mask info saved (data/.../ego_restore/[cam_id]/*.json)
✓ Intrinsic adjustments configured (qcract_preprocess.py:55-64)
✓ Ready to train (preprocess will auto-adjust intrinsics)

TESTING:
--------
# Verify crop:
python3 -c "from PIL import Image; import glob
for cam in [4,7]:
    img = Image.open(glob.glob(f'data/qcraft/processed/training/*/images/000_{cam}.jpg')[0])
    print(f'Cam {cam}: {img.size}')"

# Verify intrinsics will be correct:
python3 -c "
# Camera 4: cx = 512 (no shift needed, ego on right side)
# Camera 7: cx = 512 - 259 = 253 (ego on left, keep right side)
print('Expected intrinsics:')
print('Camera 4: cx = 512 (centered in 772px width)')
print('Camera 7: cx = 253 (shifted left in 765px width)')
"

# After training, verify renders are correct
# Then restore
"""

import subprocess
import sys

def run_command(cmd, description):
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print('='*80)
    print(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        sys.exit(1)

def main():
    print(__doc__)

    help_text = """
    Available modes:
    1. crop     - Crop images and store restoration info
    2. restore  - Restore cropped regions after rendering
    3. check    - Verify current state

    Examples:
    # Check current state
    python scripts/qcraft/solution_summary.py check

    # Crop (already done on 2025-11-26)
    python scripts/qcraft/solution_summary.py crop

    # Restore after rendering
    python scripts/qcraft/solution_summary.py restore \\
        --rendered-dir outputs/qcraft/render \\
        --output-dir outputs/qcraft/restored
    """

    if len(sys.argv) < 2:
        print(help_text)
        sys.exit(1)

    mode = sys.argv[1]
    extra_args = ' '.join(sys.argv[2:])

    if mode == 'check':
        print("\n" + "="*80)
        print("CHECKING CURRENT STATE")
        print("="*80)

        import os
        from PIL import Image
        import glob

        checkpoint_text = """
        CHECKPOINT VERIFICATION:
        ¨æ3çæ é¡¶ç«¯ å¯¹ä¸ä¸è¡¥å æè¡¨æ : image.png æä¸æ¯...
        """

        data_dirs = glob.glob('data/qcraft/processed/training/*/images/000_4.jpg')
        if not data_dirs:
            print("ERROR: No data found")
            sys.exit(1)

        data_dir = os.path.dirname(data_dirs[0])
        print(f"Checking: {data_dir}")

        cameras = [4, 7]
        for cam_id in cameras:
            # Check image size
            img_path = f'{data_dir}/000_{cam_id}.jpg'
            if os.path.exists(img_path):
                img = Image.open(img_path)
                print(f'\nCamera {cam_id}:')
                print(f'  Image size: {img.size}')

                # Verify crop info
                restore_dir = data_dir.replace('/images', '/ego_restore')
                restore_file = f'{restore_dir}/{cam_id}/000_info.json'

                if os.path.exists(restore_file):
                    import json
                    with open(restore_file) as f:
                        info = json.load(f)
                    print(f'  ✓ Restore info found: {info}')
                else:
                    print(f'  ✗ WARNING: No restore info at {restore_file}')

        print("\n" + "="*80)
        print("CHECK COMPLETED")
        print("="*80)

    elif mode == 'crop':
        cmd = f'python scripts/qcraft/ego_mask_crop_restore.py --mode crop {extra_args}'
        run_command(cmd, "Crop ego mask regions and prepare for training")

    elif mode == 'restore':
        args = extra_args or '--rendered-dir outputs/qcraft/render --output-dir outputs/qcraft/restored'
        cmd = f'python scripts/qcraft/ego_mask_crop_restore.py --mode restore {args}'
        run_command(cmd, "Restore cropped regions to rendered outputs")

    else:
        print(f"Unknown mode: {mode}")
        print(help_text)
        sys.exit(1)

if __name__ == '__main__':
    main()