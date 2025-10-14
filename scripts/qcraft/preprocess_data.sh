# 参数设置
################################################################################
scene_id="20250702_133223_Q2517"
################################################################################

export PYTHONPATH=$(pwd)
python datasets/preprocess.py \
    --data_root data/qcraft/raw/ \
    --target_dir data/qcraft/processed \
    --dataset qcraft \
    --split training \
    --scene_ids $scene_id \
    --workers 2 \
    --process_keys images lidar calib pose dynamic_masks objects