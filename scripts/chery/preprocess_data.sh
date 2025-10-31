# 参数设置
################################################################################
scene_id="clip_1746581703000"
################################################################################

export PYTHONPATH=$(pwd)
python datasets/preprocess.py \
    --data_root data/chery/raw/ \
    --target_dir data/chery/processed \
    --dataset chery \
    --split training \
    --scene_ids $scene_id \
    --workers 2 \
    --process_keys images lidar calib pose dynamic_masks objects lidar_velocities