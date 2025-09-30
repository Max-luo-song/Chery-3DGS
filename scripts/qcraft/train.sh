# 参数设置
################################################################################
cuda_device_id=7

lidar_type="lidar"  # lidar（运动补偿前）/visual（纯视觉）
config_file="configs/0908omnire_chery_lidar_wo_depth_loss.yaml"
# config_file="configs/0915omnire_chery_lidar_depth_loss.yaml"
dataset_config="qcraft/11cams_lidar"
extra_config_info=""  # 额外信息，如 depth_loss

scene_idx="20250702_133223_Q2517"

camera_ids=(0 1 2 3 4 5 6 8 9 10 12)  # camera IDs to use, e.g., (0), (0 2 4)

start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
################################################################################


output_root="output"

project_name="qcraft_${scene_idx}"

date_str=$(date +%Y%m%d)
run_name="${date_str}_${lidar_type}+cam$(IFS="_"; echo "${camera_ids[*]}")${extra_config_info}"

run_dir="${output_root}/${project_name}/${run_name}"

# 检查输出目录是否存在，防止覆盖
if [ -d "$run_dir" ]; then
    echo "错误：目录 $run_dir 已存在！请更改 run_name 或删除现有目录以避免覆盖。"
    exit 1
fi
mkdir -p "$run_dir"

# 备份当前脚本到输出目录
script_name=$(basename "$0")
backup_path="${run_dir}/${script_name}"
cp "$0" "$backup_path"

# 启动训练
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=$cuda_device_id python tools/train.py \
    --config_file $config_file \
    --output_root $output_root \
    --project $project_name \
    --run_name $run_name \
    dataset=$dataset_config \
    data.scene_idx=$scene_idx \
    data.pixel_source.cameras="[$(IFS=','; echo "${camera_ids[*]}")]" \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep