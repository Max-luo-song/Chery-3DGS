# 参数设置
################################################################################
scene_id="20250702_133223_Q2517_60_75"
################################################################################

export PYTHONPATH=$(pwd)
python datasets/preprocess.py \
    --data_root data/qcraft/raw/ \
    --target_dir data/qcraft/processed \
    --dataset qcraft \
    --split training \
    --scene_ids $scene_id \
    --workers 2 \
    --process_keys ego_masks images calib pose objects dynamic_masks lidar \
    # --skip_front_wide_side_cameras