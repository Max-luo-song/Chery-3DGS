export PYTHONPATH=$(pwd)

python datasets/preprocess.py \
    --data_root data/chery/raw/ \
    --target_dir data/chery/processed \
    --dataset chery \
    --split training \
    --split_file data/chery_scenes.txt \
    --workers 64 \
    --process_keys images lidar calib pose dynamic_masks objects lidar_velocities