import argparse
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from omegaconf import OmegaConf
from torch import Tensor
from tqdm import tqdm

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str


def move_tensors_to_cuda(data: Dict):
    for key, value in data.items():
        if isinstance(value, Tensor):
            data[key] = value.cuda(non_blocking=True)
    return data


def parse_camera_ids(camera_ids: Optional[str]) -> Optional[List[int]]:
    if camera_ids is None or camera_ids.strip() == "":
        return None
    tokens = [
        token
        for token in re.split(r"[\s,;_]+", camera_ids.replace("[", "").replace("]", ""))
        if token
    ]
    parsed = []
    for token in tokens:
        if token.lower().startswith("cam"):
            token = token[3:]
        parsed.append(int(token))
    return parsed


def build_indices(candidates: List[int], num_frames: int) -> Iterable[int]:
    if num_frames <= 0 or num_frames >= len(candidates):
        return candidates
    return candidates[:num_frames]


def get_camera_filtered_indices(dataset, camera_ids: Optional[List[int]]) -> List[int]:
    split_indices = list(dataset.split_indices)
    if camera_ids is None:
        return list(range(len(split_indices)))

    camera_data = dataset.datasource.camera_data
    missing = [cam_id for cam_id in camera_ids if cam_id not in camera_data]
    if missing:
        available = sorted(camera_data.keys())
        raise ValueError(f"Camera ids {missing} are not available. Available cameras: {available}")

    selected_unique_cam_indices = {
        int(camera_data[cam_id].unique_cam_idx) for cam_id in camera_ids
    }
    return [
        pos
        for pos, image_idx in enumerate(split_indices)
        if dataset.datasource.parse_img_idx(int(image_idx))[0] in selected_unique_cam_indices
    ]


def get_image_set(dataset, split: str):
    if split == "full":
        return dataset.full_image_set
    if split == "train":
        return dataset.train_image_set
    if split == "test":
        if dataset.test_image_set is None:
            raise ValueError("test split is not available for this dataset/config.")
        return dataset.test_image_set
    raise ValueError(f"Unsupported split: {split}")


def summarize_result(
    image_set,
    camera_ids: Optional[List[int]],
    measured: int,
    render_time: float,
    e2e_time: float,
) -> Dict:
    return {
        "camera_ids": camera_ids if camera_ids is not None else "all",
        "candidate_images": len(get_camera_filtered_indices(image_set, camera_ids)),
        "measured_images": measured,
        "render_time_sec": render_time,
        "render_fps": measured / render_time,
        "end_to_end_time_sec": e2e_time,
        "end_to_end_fps": measured / e2e_time,
    }


@torch.no_grad()
def benchmark(
    trainer,
    dataset,
    num_frames: int,
    warmup_frames: int,
    camera_ids: Optional[List[int]],
) -> Tuple[int, float, float]:
    trainer.set_eval()
    camera_downscale = trainer._get_downscale_factor()
    candidate_indices = get_camera_filtered_indices(dataset, camera_ids)
    indices = list(build_indices(candidate_indices, num_frames))
    if len(indices) == 0:
        raise ValueError("No images available for FPS benchmark.")

    total_render_time = 0.0
    total_e2e_time = 0.0
    measured = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    for step, image_idx in enumerate(tqdm(indices, desc="benchmark render", dynamic_ncols=True)):
        e2e_start = time.perf_counter()

        image_infos, cam_infos = dataset.get_image(image_idx, camera_downscale)
        image_infos = move_tensors_to_cuda(image_infos)
        cam_infos = move_tensors_to_cuda(cam_infos)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            render_start = time.perf_counter()
            _ = trainer(image_infos, cam_infos)
            torch.cuda.synchronize()
            render_time = time.perf_counter() - render_start
        else:
            render_start = time.perf_counter()
            _ = trainer(image_infos, cam_infos)
            render_time = time.perf_counter() - render_start

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e2e_time = time.perf_counter() - e2e_start

        if step >= warmup_frames:
            total_render_time += render_time
            total_e2e_time += e2e_time
            measured += 1

    if measured == 0:
        raise ValueError(
            f"warmup_frames ({warmup_frames}) must be smaller than benchmark frames ({len(indices)})."
        )

    return measured, total_render_time, total_e2e_time


def main(args):
    ckpt_path = os.path.abspath(args.resume_from)
    log_dir = os.path.dirname(ckpt_path)
    cfg_path = os.path.join(log_dir, "config.yaml")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found next to checkpoint: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.aabb.reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=ckpt_path, load_only_model=True)

    split = args.split
    image_set = get_image_set(dataset, split)

    if args.all_cameras:
        all_results = []
        camera_ids_to_benchmark = list(image_set.datasource.camera_data.keys())
        for cam_id in camera_ids_to_benchmark:
            print("")
            print(f"===== Benchmark Camera {cam_id} =====")
            measured, render_time, e2e_time = benchmark(
                trainer=trainer,
                dataset=image_set,
                num_frames=args.num_frames,
                warmup_frames=args.warmup_frames,
                camera_ids=[cam_id],
            )
            result = summarize_result(
                image_set=image_set,
                camera_ids=[cam_id],
                measured=measured,
                render_time=render_time,
                e2e_time=e2e_time,
            )
            all_results.append((cam_id, result))
            print(
                f"camera_id: {cam_id}, render_fps: {result['render_fps']:.4f}, "
                f"measured_images: {result['measured_images']}"
            )

        print("")
        print("===== Per-Camera Render FPS Summary =====")
        print("camera_id\trender_fps")
        for cam_id, result in all_results:
            print(f"{cam_id}\t{result['render_fps']:.4f}")
        return

    camera_ids = parse_camera_ids(args.camera_ids)
    measured, render_time, e2e_time = benchmark(
        trainer=trainer,
        dataset=image_set,
        num_frames=args.num_frames,
        warmup_frames=args.warmup_frames,
        camera_ids=camera_ids,
    )
    result = summarize_result(
        image_set=image_set,
        camera_ids=camera_ids,
        measured=measured,
        render_time=render_time,
        e2e_time=e2e_time,
    )
    print("")
    print("===== FPS Result =====")
    print(f"checkpoint: {ckpt_path}")
    print(f"config: {cfg_path}")
    print(f"split: {split}")
    print(f"camera_ids: {result['camera_ids']}")
    print(f"candidate_images: {result['candidate_images']}")
    print(f"measured_images: {result['measured_images']}")
    print(f"warmup_images: {args.warmup_frames}")
    print(f"render_time_sec: {result['render_time_sec']:.6f}")
    print(f"render_fps: {result['render_fps']:.4f}")
    print(f"end_to_end_time_sec: {result['end_to_end_time_sec']:.6f}")
    print(f"end_to_end_fps: {result['end_to_end_fps']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Benchmark model rendering FPS for a trained checkpoint.")
    parser.add_argument("--resume_from", required=True, type=str, help="Path to checkpoint.")
    parser.add_argument("--split", default="full", choices=["full", "train", "test"])
    parser.add_argument(
        "--camera_ids",
        default=None,
        type=str,
        help='Optional camera ids to benchmark, e.g. "0", "cam0", or "0,1,2". Default: all cameras.',
    )
    parser.add_argument(
        "--all_cameras",
        action="store_true",
        help="Benchmark every camera separately and print a per-camera render_fps summary.",
    )
    parser.add_argument(
        "--num_frames",
        default=0,
        type=int,
        help="Number of images to iterate, including warmup. Use <=0 for all images from the checkpoint config.",
    )
    parser.add_argument("--warmup_frames", default=20, type=int, help="Images excluded from timing.")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line.",
        default=None,
        nargs=argparse.REMAINDER,
    )
    main(parser.parse_args())
