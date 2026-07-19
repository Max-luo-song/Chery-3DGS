"""Convert a Gaussian Splatting PLY to a D3DR initializer.

The generated checkpoint is intentionally minimal.  D3DR only reads the six
``gauss_params`` tensors from an initializer checkpoint, so it does not need
optimizer state or training images to import an already-trained 3DGS asset.
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml


REQUIRED_PROPERTIES = {
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
}


def _read_header(path: Path) -> Tuple[int, List[str]]:
    """Return the byte offset and scalar vertex properties of a binary PLY."""
    properties: List[str] = []
    vertex_count = None
    in_vertex_element = False

    with path.open("rb") as file:
        first_line = file.readline().decode("ascii").strip()
        if first_line != "ply":
            raise ValueError(f"{path} is not a PLY file")

        while True:
            raw_line = file.readline()
            if not raw_line:
                raise ValueError(f"{path} has no end_header marker")
            line = raw_line.decode("ascii").strip()
            if line == "end_header":
                return file.tell(), properties
            if line == "format binary_little_endian 1.0":
                continue
            if line.startswith("format "):
                raise ValueError("Only binary_little_endian PLY files are supported")
            if line.startswith("element "):
                _, element_name, count = line.split()
                in_vertex_element = element_name == "vertex"
                if in_vertex_element:
                    vertex_count = int(count)
                continue
            if in_vertex_element and line.startswith("property "):
                fields = line.split()
                if len(fields) != 3 or fields[1] != "float":
                    raise ValueError("The vertex PLY properties must be scalar floats")
                properties.append(fields[2])

    # Kept for type checkers; all valid headers return at end_header.
    if vertex_count is None:
        raise ValueError("PLY contains no vertex element")


def load_gaussians(path: Path, sh_layout: str) -> dict:
    offset, properties = _read_header(path)
    missing = REQUIRED_PROPERTIES.difference(properties)
    if missing:
        raise ValueError(f"PLY is missing required Gaussian fields: {sorted(missing)}")

    rest_properties = sorted(
        (name for name in properties if name.startswith("f_rest_")),
        key=lambda name: int(name[len("f_rest_") :]),
    )
    if len(rest_properties) != 45:
        raise ValueError(
            "D3DR currently requires degree-3 SH colors (45 f_rest fields); "
            f"found {len(rest_properties)}"
        )

    with path.open("rb") as file:
        file.seek(offset)
        # PLY's vertex count is inferred from the payload.  This avoids storing
        # another header parser state and keeps the reader limited to the known
        # float-only Gaussian PLY format.
        dtype = np.dtype([(name, "<f4") for name in properties])
        values = np.fromfile(file, dtype=dtype)

    if len(values) == 0:
        raise ValueError("PLY contains no Gaussian vertices")

    def fields(names: List[str]) -> torch.Tensor:
        return torch.from_numpy(np.stack([values[name] for name in names], axis=1).copy())

    flattened_rest = fields(rest_properties)
    if sh_layout == "graphdeco":
        # Graphdeco writes its internal [N, 15, 3] tensor after first
        # transposing it to [N, 3, 15], then flattening it.
        features_rest = flattened_rest.reshape(-1, 3, 15).transpose(1, 2)
    elif sh_layout == "direct":
        # scene_reconstruction-main's export_ply.py flattens its internal
        # [N, 15, 3] tensor directly, without a transpose.
        features_rest = flattened_rest.reshape(-1, 15, 3)
    else:
        raise ValueError(f"Unsupported SH layout: {sh_layout}")
    result = {
        "means": fields(["x", "y", "z"]),
        "features_dc": fields(["f_dc_0", "f_dc_1", "f_dc_2"]),
        "features_rest": features_rest,
        "opacities": fields(["opacity"]),
        "scales": fields(["scale_0", "scale_1", "scale_2"]),
        "quats": fields(["rot_0", "rot_1", "rot_2", "rot_3"]),
    }
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("PLY contains non-finite Gaussian parameters")
    return result


def write_render_config(template_path: Path, output_dir: Path, data_path: Path) -> None:
    """Create a config whose checkpoint directory is ``output_dir/nerfstudio_models``."""
    if len(output_dir.parents) < 3:
        raise ValueError("--output-dir must have the layout root/experiment/method/timestamp")

    config = yaml.load(template_path.read_text(), Loader=yaml.Loader)
    config.output_dir = output_dir.parents[2]
    config.experiment_name = output_dir.parents[1].name
    config.method_name = output_dir.parent.name
    config.timestamp = output_dir.name
    config.load_dir = None
    config.load_step = None
    config.load_checkpoint = None
    config.data = data_path
    config.pipeline.datamanager.data = data_path
    expected_checkpoint_dir = config.get_checkpoint_dir()
    checkpoint_dir = output_dir / "nerfstudio_models"
    if expected_checkpoint_dir != checkpoint_dir:
        raise RuntimeError(
            f"Template configuration resolves to {expected_checkpoint_dir}, expected {checkpoint_dir}"
        )
    (output_dir / "config.yml").write_text(yaml.dump(config))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument(
        "--sh-layout",
        choices=("graphdeco", "direct"),
        default="graphdeco",
        help=(
            "Order of the 45 f_rest_* properties. Use graphdeco for normal "
            "3DGS PLYs; use direct for scene_reconstruction-main export_ply.py output."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Initializer directory passed to --init_obj_path",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Nerfstudio data directory used only when rendering the imported asset",
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        help=(
            "A splatfacto config.yml from this Nerfstudio version. Supplying it makes "
            "the imported asset renderable by train_everything.py."
        ),
    )
    parser.add_argument(
        "--sparse-pc-out",
        type=Path,
        help=(
            "Optional point-cloud PLY for D3DR's object-view sampler. The input "
            "Gaussian PLY is copied because its x/y/z fields are a valid point cloud."
        ),
    )
    args = parser.parse_args()

    params = load_gaussians(args.input_ply, args.sh_layout)
    checkpoint_dir = args.output_dir / "nerfstudio_models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "step-000000000.ckpt"
    checkpoint = {
        "step": 0,
        "pipeline": {
            f"_model.gauss_params.{name}": value
            for name, value in params.items()
        },
    }
    torch.save(checkpoint, checkpoint_path)

    # D3DR applies this matrix when it combines object and scene parameters.
    # The imported PLY is already expressed in its own local coordinate system.
    (args.output_dir / "dataparser_transforms.json").write_text(
        '{\n  "transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],\n  "scale": 1.0\n}\n'
    )
    if (args.template_config is None) != (args.data_path is None):
        parser.error("--template-config and --data-path must be supplied together")
    if args.template_config is not None:
        write_render_config(args.template_config, args.output_dir, args.data_path)
    if args.sparse_pc_out is not None:
        args.sparse_pc_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.input_ply, args.sparse_pc_out)
    print(f"Imported {params['means'].shape[0]} Gaussians with SH layout: {args.sh_layout}")
    print(f"D3DR initializer: {args.output_dir}")
    print("Pass this directory with --init_obj_path and pass an explicit --transforms_obj path.")


if __name__ == "__main__":
    main()
