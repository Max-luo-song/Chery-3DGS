import torch
import argparse

def find_and_count_gaussians(ckpt_path):
    """
    探索模型属性，并尝试统计高斯点数量
    """
    print(f"正在分析 checkpoint: {ckpt_path}")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if 'models' not in checkpoint or not isinstance(checkpoint['models'], dict):
            print("错误：Checkpoint 中没有找到 'models' 字典。")
            return

        models_dict = checkpoint['models']
        total_gaussians = 0

        # 定义一个辅助函数来处理每个模型组件
        def analyze_component(component_name, state_dict):
            nonlocal total_gaussians
            print(f"\n--- 分析 {component_name} ---")
            print("可用的属性:")
            candidate_keys = []
            for i, key in enumerate(state_dict.keys()):
                # 我们只关心可能存储点坐标的张量，通常名字里会包含 xyz, points, gaussians 等
                print(f"  {i+1}. {key} (shape: {state_dict[key].shape})")
                if 'xyz' in key.lower() or 'points' in key.lower() or 'gaussians' in key.lower():
                    candidate_keys.append(key)

            if not candidate_keys:
                print(f"  -> 未在 {component_name} 中找到明显的高斯点属性。")
                return

            # 如果有多个候选，我们选择第一个，或者让用户决定（这里自动选第一个）
            key_to_check = candidate_keys[0]
            print(f"\n  -> 推测 '{key_to_check}' 可能是高斯点属性，尝试统计其数量...")
            
            points_tensor = state_dict[key_to_check]
            # 高斯点坐标张量的形状通常是 (N, 3)
            if len(points_tensor.shape) == 2 and points_tensor.shape[1] == 3:
                num_points = points_tensor.shape[0]
                print(f"  -> {component_name} 中的高斯点数量 (基于 {key_to_check}): {num_points:,}")
                return num_points
            else:
                print(f"  -> 张量 {key_to_check} 的形状 {points_tensor.shape} 不像高斯点坐标，跳过统计。")
                return 0

        # --- 分析 Background ---
        if 'Background' in models_dict:
            bg_points = analyze_component('Background', models_dict['Background'])
            if bg_points:
                total_gaussians += bg_points

        # --- 分析 RigidNodes ---
        if 'RigidNodes' in models_dict:
            rn_points = analyze_component('RigidNodes', models_dict['RigidNodes'])
            if rn_points:
                total_gaussians += rn_points

        # --- 输出总数 ---
        print("\n" + "="*40)
        print(f"Background 和 RigidNodes 中的高斯点总数: {total_gaussians:,}")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{ckpt_path}'，请检查路径是否正确。")
    except Exception as e:
        print(f"处理 checkpoint 时发生错误: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="探索并统计 checkpoint 中的高斯点数量")
    parser.add_argument("--ckpt_file", type=str, help="checkpoint 文件的路径")
    
    args = parser.parse_args()
    find_and_count_gaussians(args.ckpt_file)
