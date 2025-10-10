export PYTHONPATH=$(pwd)

scene_ids="20250702_133223_Q2517_60_75"

CUDA_VISIBLE_DEVICES=6 python datasets/preprocess.py \
    --data_root /data/baitongyao/qcraft_0702/ \
    --target_dir /data/baitongyao/processed \
    --dataset qcraft \
    --split training \
    --scene_ids $scene_ids \
    --workers 2 \
    --process_keys images lidar calib pose dynamic_masks objects