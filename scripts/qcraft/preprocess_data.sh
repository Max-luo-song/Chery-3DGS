export PYTHONPATH=$(pwd)

scene_ids="20250702_133223_Q2517"

CUDA_VISIBLE_DEVICES=6 python datasets/preprocess.py \
    --data_root data/qcraft/raw/ \
    --target_dir data/qcraft/processed \
    --dataset qcraft \
    --split training \
    --scene_ids $scene_ids \
    --workers 2 \
    --process_keys images lidar calib pose
    # --process_keys images lidar calib pose dynamic_masks objects