import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--tr_path", type=str, required=True)
parser.add_argument("--h", type=int, default=None)
parser.add_argument(
    "--opencv-to-nerfstudio",
    action="store_true",
    help="Convert OpenCV C2W poses (+Z forward, +Y down) to Nerfstudio/OpenGL C2W.",
)

parser.add_argument("--save_name", type=str, default="camera_path.json")
parser.add_argument("--seconds", type=float, default=5.0)

args = parser.parse_args()
with open(args.tr_path, "r") as f:
    data_init = json.load(f)

data_result = {}

if args.h is not None:
    data_init["h"] = args.h
    data_init["w"] = args.h

data_result["render_height"] = data_init["h"]
data_result["render_width"] = data_init["w"]
data_result["camera_type"] = "perspective"
data_result["seconds"] = args.seconds
data_result["camera_path"] = []

for f in data_init["frames"]:
    camera_to_world = np.asarray(f["transform_matrix"], dtype=np.float64)
    if args.opencv_to_nerfstudio:
        camera_to_world[:3, 1:3] *= -1
    data_result["camera_path"].append(
        {
            "camera_to_world": camera_to_world.tolist(),
            "fov": np.rad2deg(data_init["camera_angle_x"]),
        }
    )

if (
    args.save_name.startswith("/") or args.save_name.startswith("./")
) and os.path.exists(os.path.dirname(args.save_name)):
    save_path = args.save_name
else:
    save_path = os.path.join(os.path.dirname(args.tr_path), args.save_name)
with open(save_path, "w") as f:
    json.dump(data_result, f, indent=4)
print("Done transform_2_camera_path!")
