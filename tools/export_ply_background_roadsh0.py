import os
import os.path as osp
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import argparse


class Gaussian:
    def __init__(self, gaussian):
        self._means = gaussian["_means"].cpu().numpy()
        self._features_dc = gaussian["_features_dc"].cpu().numpy()
        self._features_rest = gaussian["_features_rest"].cpu().numpy() if "_features_rest" in gaussian else None
        self._opacities = gaussian["_opacities"].cpu().numpy()
        self._scales = gaussian["_scales"].cpu().numpy()
        self._quats = gaussian["_quats"].cpu().numpy()


def export_ply(pth_path, out_path):
    data = torch.load(pth_path)

    gaussian = Gaussian(data["models"]["Background"])
    gaussian_road = Gaussian(data["models"]["RoadNodes"])
    
    # Background has 3rd order SH (1 DC + 15 rest = 16 coefficients)
    # RoadNodes has 0th order SH (1 DC only)
    
    # merge road nodes into background gaussians
    # 合并 road nodes into background gaussians
    # 依次合并各个属性张量
    xyz = np.concatenate([gaussian._means, gaussian_road._means], axis=0)
    
    normals = np.zeros_like(xyz)
    
    # 合并 DC 特征
    f_dc = np.concatenate([gaussian._features_dc, gaussian_road._features_dc], axis=0)
    f_dc = f_dc.reshape((f_dc.shape[0], -1))
    
    # 合并 Rest 特征 - RoadNodes 需要补零
    if gaussian_road._features_rest is None:
        # RoadNodes 没有 Rest 特征，创建全零数组
        # 3阶SH的Rest部分有15个系数，每个系数有3个颜色通道
        road_f_rest = np.zeros((gaussian_road._means.shape[0], 15, 3))
    else:
        road_f_rest = gaussian_road._features_rest
    
    # 确保两个数组的形状一致
    if gaussian._features_rest.shape[1] != road_f_rest.shape[1]:
        # 如果维度不匹配，需要调整RoadNodes的维度
        # 这里假设RoadNodes是0阶SH，需要扩展到3阶SH的维度
        current_rest_dim = road_f_rest.shape[1]
        target_rest_dim = gaussian._features_rest.shape[1]
        if current_rest_dim < target_rest_dim:
            # 补零到目标维度
            padding = np.zeros((road_f_rest.shape[0], target_rest_dim - current_rest_dim, 3))
            road_f_rest = np.concatenate([road_f_rest, padding], axis=1)
        elif current_rest_dim > target_rest_dim:
            # 截断到目标维度（通常不会发生）
            road_f_rest = road_f_rest[:, :target_rest_dim, :]
    
    f_rest = np.concatenate([gaussian._features_rest, road_f_rest], axis=0)
    f_rest = f_rest.reshape((f_rest.shape[0], -1))
    
    opacities = np.concatenate([gaussian._opacities, gaussian_road._opacities], axis=0)
    
    scale = np.concatenate([gaussian._scales, gaussian_road._scales], axis=0)
    
    rotation = np.concatenate([gaussian._quats, gaussian_road._quats], axis=0)

    def construct_list_of_attributes(gaussian):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(3):
            l.append("f_dc_{}".format(i))
        
        # Rest features - for 3rd order SH, there are 15 * 3 = 45 features
        rest_features_count = f_rest.shape[1]  # This should be 45 for 3rd order SH
        for i in range(rest_features_count):
            l.append("f_rest_{}".format(i))
            
        l.append("opacity")
        for i in range(gaussian._scales.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(gaussian._quats.shape[1]):
            l.append("rot_{}".format(i))
        return l

    dtype_full = [
        (attribute, "f4") for attribute in construct_list_of_attributes(gaussian)
    ]
    
    # 确保所有数组的形状正确
    # f_dc 应该是 [N, 3]
    # f_rest 应该是 [N, 45] (3阶SH)
    # scale 应该是 [N, 3]
    # rotation 应该是 [N, 4]
    
    # 打印调试信息
    print(f"xyz shape: {xyz.shape}")
    print(f"f_dc shape: {f_dc.shape}")
    print(f"f_rest shape: {f_rest.shape}")
    print(f"opacities shape: {opacities.shape}")
    print(f"scale shape: {scale.shape}")
    print(f"rotation shape: {rotation.shape}")
    
    # 合并所有属性
    attributes = np.concatenate([xyz, normals, f_dc, f_rest, opacities, scale, rotation], axis=1)
    
    # 验证维度
    expected_dim = 3 + 3 + 3 + f_rest.shape[1] + 1 + scale.shape[1] + rotation.shape[1]
    actual_dim = attributes.shape[1]
    print(f"Expected dimension: {expected_dim}, Actual dimension: {actual_dim}")
    
    if expected_dim != actual_dim:
        print("Warning: Dimension mismatch! Check the feature dimensions.")
    
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(out_path)
    print(f"PLY file saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export background gaussians to ply file")

    # misc
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()

    run_dir = os.path.dirname(args.ckpt_path)

    save_dir = osp.join(run_dir, "point_clouds")
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, "background.ply")

    export_ply(args.ckpt_path, save_path)