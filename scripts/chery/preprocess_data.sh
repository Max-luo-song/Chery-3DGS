export PYTHONPATH=$(pwd)

scene_ids="clip_1746752396800"

python datasets/preprocess.py \
    --data_root data/chery/raw/ \
    --target_dir data/chery/processed \
    --dataset chery \
    --split training \
    --scene_ids $scene_ids \
    --workers 2 \
    --process_keys images lidar calib pose dynamic_masks objects lidar_velocities