SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/utils.sh

gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
  echo "no gpu found"
  exit 1
fi

export PYTHONPATH=$(pwd)

project="chery_clip_1746752396800"
run_name="20250911_mclidar+cam0+depth_loss"

CUDA_VISIBLE_DEVICES=${gpu} python chery_tools/export_ply.py \
    --project $project \
    --run_name $run_name