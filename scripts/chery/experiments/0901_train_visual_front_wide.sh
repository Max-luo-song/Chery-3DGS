export PYTHONPATH=$(pwd)

start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

CUDA_VISIBLE_DEVICES=5 python tools/train.py \
    --config_file configs/0901omnire_chery_visual.yaml \
    --output_root output \
    --project omnire \
    --run_name 0901_chery_visual_front_wide \
    dataset=chery/1cams_visual \
    data.scene_idx=clip_1746752396800 \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep