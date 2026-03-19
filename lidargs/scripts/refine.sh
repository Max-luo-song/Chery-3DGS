iterations=10000
gpu=0
# 预处理之后的文件夹
#data="/home/not0513/data/qcraft/processed/training/20251025_163358_QCOYSD504206_1595_1610"
#caseid="20251025_163358_QCOYSD504206_1595_1610"
# 模型文件路径，参考data目录，training改为test，再按照caseid/日期+时间自动生成output_dir
#test_str="/home/not0513/data/qcraft/processed/test"
#date_str="2026-01-06-11-06"
#output_dir="${test_str}/${caseid}/${date_str}"
# 预处理之后的文件夹
#/nas_thoru/oldbak/yx/datavtd/lijiaoqiao_20260205_02_offset_0.0m
#20251121_095534_QCOYPDA27344_1538_1553
#20251025_163358_QCOYSD504206_1595_1610
#data="/nas_thoru/oldbak/yx/1210new/scene_reconstruction-users-yangtao-fix_block_id/data/qcraft/processed/training/20251025_163358_QCOYSD504206_1595_1610"
data="/nas_thoru/oldbak/yx/datavtd/lijiaoqiao_20260205_02_offset_0.0m"
caseid="lijiaoqiao_20260205_02_offset_0.0m"
# 模型文件路径，参考data目录，training改为test，再按照caseid/日期+时间自动生成output_dir
test_str="/nas_thoru/oldbak/yx/M3/vtdoutput/yxoutput312"
date_str="2026-03-12"
output_dir="${test_str}/${caseid}/${date_str}"

# 将项目根目录加入 PYTHONPATH(上两级)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

enable_raydrop_unet=true

if ${enable_raydrop_unet}; then
    ENABLE_RAYDROP_UNET="--enable_raydrop_unet"
    echo "Enable Raydrop UNet."
else
    ENABLE_RAYDROP_UNET=""
    echo "Raydrop UNet disabled."
fi

if ${enable_raydrop_unet}; then
    python3 $PROJECT_ROOT/lidargs/refine_ray_drop.py -s ${data} -m ${output_dir} \
                --gpu ${gpu} --dataset vtd --block_size 50
fi