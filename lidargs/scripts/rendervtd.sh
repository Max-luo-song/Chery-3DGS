iterations=10000
data="/nas_thoru/oldbak/yx/datavtd/lijiaoqiao_20260205_02_offset_0.0m"
caseid="lijiaoqiao_20260205_02_offset_0.0m"
test_str="/nas_thoru/oldbak/yx/M3/vtdoutput/yxoutput312"
date_str="2026-03-12"

output_dir="${test_str}/${caseid}/${date_str}"
block_size=50
output_path="${output_dir}/render"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python3 $PROJECT_ROOT/lidargs/render.py \
    -s "${data}" \
    -m "${output_dir}" \
    --caseid "${caseid}" \
    --iteration "${iterations}" \
    --max_depth 150 \
    --block_size "${block_size}" \
    --output_path "${output_path}"\
    --dataset "vtd" \
    --test_frames 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48