# 参数设置
################################################################################
scene_id="20251103_134932_QLC0N1000623"
################################################################################

export PYTHONPATH=$(pwd)
python datasets/preprocess_qcraft.py \
    --data_root data/qcraft/raw/ \
    --target_dir data/qcraft/processed \
    --dataset qcraft \
    --split training \
    --scene_ids $scene_id \
    --workers 2 \
    --process_keys images lidar calib pose dynamic_masks objects