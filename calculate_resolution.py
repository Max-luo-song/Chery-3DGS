import numpy as np
from PIL import Image

def calculate_black_resolution(image_path):
    """
    计算图片中黑色部分的分辨率
    """
    # 读取图片
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # 如果是彩色图片，转换为灰度
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)
    
    # 找到黑色区域（假设黑色像素值接近0）
    black_mask = img_array < 50  # 阈值设为50，可以根据实际情况调整
    
    # 找到黑色区域的边界
    black_rows = np.any(black_mask, axis=1)
    black_cols = np.any(black_mask, axis=0)
    
    # 获取黑色区域的边界坐标
    top = np.argmax(black_rows)
    bottom = len(black_rows) - np.argmax(black_rows[::-1])
    left = np.argmax(black_cols)
    right = len(black_cols) - np.argmax(black_cols[::-1])
    
    # 计算黑色区域的分辨率
    black_width = right - left
    black_height = bottom - top
    
    print(f"图片总分辨率: {img_array.shape[1]} x {img_array.shape[0]}")
    print(f"黑色区域分辨率: {black_width} x {black_height}")
    print(f"黑色区域位置: 左上角({left}, {top}) 到 右下角({right}, {bottom})")
    
    return black_width, black_height

# 使用示例
image_path = "data/qcraft/processed/training/20251025_163358_QCOYSD504206/ego_masks/1.png"
width, height = calculate_black_resolution(image_path)
