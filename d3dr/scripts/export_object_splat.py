#!/usr/bin/env python3
"""Write D3DR's learned object SH colours back into the original object PLY.

The output retains the geometric parameters from ``--source-ply`` (means,
scales and rotations) in the original local coordinate system.  Its degree-3
spherical-harmonic fields and opacity are replaced with the learned D3DR
object fields.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument(
        "--source-ply",
        type=Path,
        required=True,
        help="The original, local-coordinate Graphdeco 3DGS PLY.",
    )
    parser.add_argument(
        "--output-ply",
        type=Path,
        required=True,
        help="Output PLY with the original geometry and learned SH fields.",
    )
    return parser.parse_args()


def read_ply(path: Path):
    """Read a float-only binary Graphdeco PLY, preserving its header verbatim."""
    properties = []
    with path.open("rb") as file:
        header = bytearray()
        in_vertex = False
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"{path} has no end_header marker")
            header.extend(line)
            text = line.decode("ascii").strip()
            if text == "end_header":
                break
            if text.startswith("element "):
                in_vertex = text.split()[1] == "vertex"
            elif in_vertex and text.startswith("property "):
                parts = text.split()
                if len(parts) != 3 or parts[1] != "float":
                    raise ValueError("Only float-only Gaussian PLY files are supported")
                properties.append(parts[2])
        values = np.fromfile(file, dtype=np.dtype([(name, "<f4") for name in properties]))
    return bytes(header), properties, values


def find_checkpoint(config_path: Path) -> Path:
    """Find the latest checkpoint next to a Nerfstudio config.yml."""
    checkpoint_dir = config_path.parent / "nerfstudio_models"
    checkpoints = sorted(checkpoint_dir.glob("step-*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    return checkpoints[-1]


def parameter(pipeline_state: dict, name: str) -> torch.Tensor:
    suffix = f"gauss_params.{name}"
    matches = [value for key, value in pipeline_state.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one checkpoint field ending in {suffix}")
    return matches[0]


def main() -> None:
    args = parse_args()
    header, properties, source = read_ply(args.source_ply)
    checkpoint_path = find_checkpoint(args.load_config)
    print(f"Loading checkpoint directly: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    pipeline_state = checkpoint["pipeline"]

    # D3DR concatenates [scene, object] and this run disables densification,
    # so the last N records correspond exactly to the original N object splats.
    object_count = len(source)
    features_dc = parameter(pipeline_state, "features_dc")
    features_rest = parameter(pipeline_state, "features_rest")
    raw_opacities = parameter(pipeline_state, "opacities")
    if min(features_dc.shape[0], features_rest.shape[0], raw_opacities.shape[0]) < object_count:
        raise RuntimeError(
            f"Source PLY has {object_count} Gaussians, more than this checkpoint."
        )
    shs_0 = features_dc[-object_count:].detach().cpu().numpy()
    shs_rest = features_rest[-object_count:].transpose(1, 2).contiguous().cpu().numpy()
    shs_rest = shs_rest.reshape((object_count, -1))
    # Graphdeco PLY stores opacity logits, so retain the raw checkpoint value.
    opacities = raw_opacities[-object_count:].detach().cpu().numpy().reshape(-1)
    if shs_0.ndim == 3:
        shs_0 = shs_0.squeeze(-1)
    if shs_0.shape != (object_count, 3):
        raise RuntimeError(f"Expected DC SH shape ({object_count}, 3), got {shs_0.shape}")
    dc_fields = [f"f_dc_{i}" for i in range(3)]
    rest_fields = [f"f_rest_{i}" for i in range(shs_rest.shape[1])]
    missing = [field for field in dc_fields + rest_fields + ["opacity"] if field not in properties]
    if missing:
        raise RuntimeError(f"Source PLY is missing SH fields: {missing}")

    result = source.copy()
    for index, field in enumerate(dc_fields):
        result[field] = shs_0[:, index]
    for index, field in enumerate(rest_fields):
        result[field] = shs_rest[:, index]
    result["opacity"] = opacities

    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    with args.output_ply.open("wb") as file:
        file.write(header)
        result.tofile(file)
    print(f"Wrote {len(result)} local-coordinate object Gaussians to {args.output_ply}")


if __name__ == "__main__":
    main()
