export PYTHONPATH=$(pwd)

project="chery_clip_1746752396800"
run_name="20250911_mclidar+cam0+depth_loss"

CUDA_VISIBLE_DEVICES=5 python chery_tools/export_ply.py \
    --project $project \
    --run_name $run_name