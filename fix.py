import torch
import os

def fix_cuda_device():
    print("=== CUDA Device Status ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return "cpu"
    
    try:
        # 方法1: 设置设备
        torch.cuda.set_device(0)
        current_device = torch.cuda.current_device()
        print(f"Current device after set_device: {current_device}")
        
        # 方法2: 测试GPU计算
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x + y
        print(f"GPU computation test passed: {z.shape}")
        
        # 方法3: 清理缓存
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
        
        return f"cuda:{current_device}"
        
    except Exception as e:
        print(f"GPU test failed: {e}")
        print("Falling back to CPU")
        return "cpu"

if __name__ == "__main__":
    device = fix_cuda_device()
    print(f"Final device: {device}")
