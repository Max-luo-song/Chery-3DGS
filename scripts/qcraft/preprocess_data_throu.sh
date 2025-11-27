# 参数设置
################################################################################
scene_id="20251025_163358_QCOYSD504206"
################################################################################

export PYTHONPATH=$(pwd)
python datasets/preprocess.py \
    --data_root data/qcraft/raw/ \
    --target_dir data/qcraft/processed \
    --dataset qcraft \
    --split training \
    --scene_ids $scene_id \
    --workers 2 \
    --process_keys images calib dynamic_masks objects


#     --process_keys images lidar calib pose dynamic_masks objects