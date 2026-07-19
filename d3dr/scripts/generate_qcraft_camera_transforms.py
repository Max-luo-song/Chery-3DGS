#!/usr/bin/env python3
"""Create D3DR transforms.json from visible QCraft ego-camera poses.

The output keeps only frames where the configured object centre is actually
inside the selected camera's image.  Camera intrinsics/extrinsics are read
directly from the scene_reconstruction-main QCraft processed dataset.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qcraft-root", type=Path, required=True)
    parser.add_argument("--camera-id", type=int, required=True)
    parser.add_argument("--transforms-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--min-distance", type=float, default=2.0)
    parser.add_argument("--max-distance", type=float, default=30.0)
    parser.add_argument("--visibility-margin", type=float, default=0.05)
    return parser.parse_args()


def camera_to_worlds(root: Path, camera_id: int) -> list:
    lidar_poses = sorted((root / "lidar_pose").glob("*.txt"))
    if not lidar_poses:
        raise FileNotFoundError(f"No lidar poses in {root / 'lidar_pose'}")
    first_pose = np.loadtxt(lidar_poses[0])
    cam_to_lidar = np.loadtxt(root / "extrinsics" / f"{camera_id}.txt")
    return [np.linalg.inv(first_pose) @ np.loadtxt(pose) @ cam_to_lidar for pose in lidar_poses]


def visible_frame_ids(poses: list, center: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                      width: int, height: int, min_distance: float, max_distance: float, margin: float) -> list:
    ids = []
    for index, pose in enumerate(poses):
        # QCraft stores OpenCV C2W (+Z forward, +Y down), which is the convention
        # used in the dataset calibration files.
        point_camera = pose[:3, :3].T @ (center - pose[:3, 3])
        distance = np.linalg.norm(point_camera)
        if point_camera[2] <= 0 or not min_distance <= distance <= max_distance:
            continue
        u = fx * point_camera[0] / point_camera[2] + cx
        v = fy * point_camera[1] / point_camera[2] + cy
        if -margin * width <= u <= (1 + margin) * width and -margin * height <= v <= (1 + margin) * height:
            ids.append(index)
    return ids


def main() -> None:
    args = parse_args()
    intrinsics = np.loadtxt(args.qcraft_root / "intrinsics" / f"{args.camera_id}.txt")
    fx, fy, cx, cy = map(float, intrinsics[:4])
    distortion = list(map(float, intrinsics[4:8]))
    image_path = args.qcraft_root / "images" / f"000000_{args.camera_id}.png"
    try:
        from PIL import Image
        width, height = Image.open(image_path).size
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Cannot determine camera resolution from {image_path}") from error

    poses = camera_to_worlds(args.qcraft_root, args.camera_id)
    for transform_dir in args.transforms_dirs:
        path = transform_dir / "transforms.json"
        data = json.loads(path.read_text())
        center = np.asarray(data["object_center"], dtype=np.float64)
        keep_ids = visible_frame_ids(
            poses, center, fx, fy, cx, cy, width, height,
            args.min_distance, args.max_distance, args.visibility_margin,
        )
        if not keep_ids:
            raise RuntimeError(f"No visible camera-{args.camera_id} pose for object centre {center.tolist()}")

        data.update(
            {
                "camera_angle_x": float(2 * np.arctan(width / (2 * fx))),
                "camera_angle_y": float(2 * np.arctan(height / (2 * fy))),
                "w": width,
                "h": height,
                "fl_x": fx,
                "fl_y": fy,
                "cx": cx,
                "cy": cy,
                "k1": distortion[0],
                "k2": distortion[1],
                "p1": distortion[2],
                "p2": distortion[3],
                "camera_model": "OPENCV",
                "frames": [
                    {
                        "file_path": f"images/{frame_id:06d}_{args.camera_id}.png",
                        "transform_matrix": poses[frame_id].tolist(),
                    }
                    for frame_id in keep_ids
                ],
            }
        )
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"{path}: kept camera-{args.camera_id} frames {keep_ids}")


if __name__ == "__main__":
    main()
