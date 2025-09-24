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
        self._features_rest = gaussian["_features_rest"].cpu().numpy()
        self._opacities = gaussian["_opacities"].cpu().numpy()
        self._scales = gaussian["_scales"].cpu().numpy()
        self._quats = gaussian["_quats"].cpu().numpy()


def export_ply(pth_path, out_path):
    data = torch.load(pth_path)

    gaussian = Gaussian(data["models"]["Background"])
    xyz = gaussian._means
    normals = np.zeros_like(xyz)
    f_dc = gaussian._features_dc.reshape((gaussian._features_dc.shape[0], -1))
    f_rest = gaussian._features_rest.reshape((gaussian._features_rest.shape[0], -1))
    opacities = gaussian._opacities
    scale = gaussian._scales
    rotation = gaussian._quats

    def construct_list_of_attributes(gaussian):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(3):
            l.append("f_dc_{}".format(i))
        for i in range(45):
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
    attribute_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(attribute_list, axis=1)
    # do not save 'features_extra' for ply
    # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, f_extra), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export background gaussians to ply file")
    
    # misc
    parser.add_argument(
        "--project",
        help="Name of the project",
        type=str,
        required=True
    )
    parser.add_argument(
        "--run_name",
        help="Name of the run",
        type=str,
        required=True
    )
    args = parser.parse_args()

    run_dir = osp.join(
        "output",
        args.project,
        args.run_name,
    )
    
    pc_dir = osp.join(run_dir, "point_clouds")
    os.makedirs(pc_dir, exist_ok=True)

    pth_path = osp.join(run_dir, "checkpoint_final.pth")
    out_path = osp.join(pc_dir, "background.ply")

    export_ply(pth_path, out_path)
