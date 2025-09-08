export PYTHONPATH=$(pwd)

start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

CUDA_VISIBLE_DEVICES=6 python tools/train.py \
    --config_file configs/0831omnire_chery_lidar.yaml \
    --output_root output \
    --project omnire \
    --run_name 0901_chery_lidar_front_main \
    dataset=chery/1cams_lidar \
    data.scene_idx=clip_1746752396800 \
    data.pixel_source.cameras=[1] \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep