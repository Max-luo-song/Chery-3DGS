#!/bin/bash

# 检查参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <ckpt_path>"
    echo "Example: $0 /path/to/checkpoint.pth"
    exit 1
fi

CKPT_PATH=$1

# 获取脚本所在目录的项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Python 脚本路径
EXPORT_ROAD_SCRIPT="$PROJECT_ROOT/tools/export_ply_road_sh0.py"
EXPORT_BACKGROUND_SCRIPT="$PROJECT_ROOT/tools/export_ply_background_roadsh0.py"

# 检查文件是否存在
if [ ! -f "$EXPORT_ROAD_SCRIPT" ]; then
    echo "Error: $EXPORT_ROAD_SCRIPT not found"
    exit 1
fi

if [ ! -f "$EXPORT_BACKGROUND_SCRIPT" ]; then
    echo "Error: $EXPORT_BACKGROUND_SCRIPT not found"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint file $CKPT_PATH not found"
    exit 1
fi

echo "================================================"
echo "Exporting PLY files from checkpoint: $CKPT_PATH"
echo "================================================"

# 导出 road ply
echo ""
echo "[1/2] Exporting road ply..."
python "$EXPORT_ROAD_SCRIPT" --ckpt_path "$CKPT_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to export road ply"
    exit 1
fi

# 导出 background ply
echo ""
echo "[2/2] Exporting background ply..."
python "$EXPORT_BACKGROUND_SCRIPT" --ckpt_path "$CKPT_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to export background ply"
    exit 1
fi

echo ""
echo "================================================"
echo "Export completed successfully!"
echo "================================================"
