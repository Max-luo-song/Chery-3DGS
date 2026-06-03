import argparse
import logging
import os
import re
import shutil
from types import SimpleNamespace
from typing import Optional

from tools import compare

logger = logging.getLogger()


def has_metrics(log_dir: str) -> bool:
    for metrics_name in ["metrics_eval", "metrics"]:
        metrics_dir = os.path.join(log_dir, metrics_name)
        if not os.path.isdir(metrics_dir):
            continue
        for name in os.listdir(metrics_dir):
            if name.startswith("images_full") and name.endswith(".json"):
                return True
    return False


def has_parallel_outputs(log_dir: str) -> bool:
    return (
        os.path.isdir(os.path.join(log_dir, "group0"))
        and os.path.isdir(os.path.join(log_dir, "group1"))
        and os.path.isdir(os.path.join(log_dir, "videos"))
        and has_metrics(log_dir)
    )


def resolve_log_dir(path: str, prefer_parallel: bool = False) -> str:
    if has_parallel_outputs(path) or has_metrics(path):
        return path

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory does not exist: {path}")

    children = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]
    if prefer_parallel:
        candidates = [child for child in children if has_parallel_outputs(child)]
    else:
        candidates = [child for child in children if has_metrics(child)]

    if not candidates:
        kind = "view-parallel run" if prefer_parallel else "run"
        raise FileNotFoundError(f"Cannot find a comparable {kind} under: {path}")

    candidates.sort(key=os.path.getmtime, reverse=True)
    resolved = candidates[0]
    logger.info(f"Resolved {path} -> {resolved}")
    return resolved


def replace_link(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        if os.path.isdir(dst) and not os.path.islink(dst):
            shutil.rmtree(dst)
        else:
            os.unlink(dst)
    os.symlink(os.path.abspath(src), dst)


def find_render_dir(videos_dir: str, suffix: str, exclude_prefix: Optional[str] = None) -> Optional[str]:
    if not os.path.isdir(videos_dir):
        return None
    candidates = []
    for name in os.listdir(videos_dir):
        path = os.path.join(videos_dir, name)
        if not os.path.isdir(path):
            continue
        if not name.endswith(suffix):
            continue
        if exclude_prefix is not None and name.startswith(exclude_prefix):
            continue
        candidates.append(path)
    if not candidates:
        return None
    exact_pattern = re.compile(rf"^full_set_\d+{re.escape(suffix)}$")
    exact_candidates = [
        path for path in candidates if exact_pattern.match(os.path.basename(path))
    ]
    if exact_candidates:
        candidates = exact_candidates
    return max(candidates, key=os.path.getmtime)


def stage_parallel_dir(log_dir: str) -> str:
    videos_dir = os.path.join(log_dir, "videos")
    rgbs_dir = find_render_dir(videos_dir, "_rgbs", exclude_prefix="gt_")
    gt_rgbs_dir = find_render_dir(videos_dir, "_gt_rgbs")
    metrics_eval_dir = os.path.join(log_dir, "metrics_eval")

    if rgbs_dir is None:
        raise FileNotFoundError(
            f"Cannot find merged rgb images under {videos_dir}. "
            "Expected a directory like videos/full_set_40000_rgbs."
        )

    stage_dir = os.path.join(log_dir, ".compare_view_parallel")
    os.makedirs(os.path.join(stage_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(stage_dir, "metrics_eval"), exist_ok=True)

    replace_link(rgbs_dir, os.path.join(stage_dir, "videos", "full_set_00000_rgbs"))
    if gt_rgbs_dir is not None and os.path.isdir(gt_rgbs_dir):
        replace_link(gt_rgbs_dir, os.path.join(stage_dir, "videos", "full_set_00000_gt_rgbs"))

    if os.path.isdir(metrics_eval_dir):
        for name in os.listdir(metrics_eval_dir):
            if name.startswith("images_full") and name.endswith(".json"):
                replace_link(
                    os.path.join(metrics_eval_dir, name),
                    os.path.join(stage_dir, "metrics_eval", name),
                )

    stage_config = os.path.join(stage_dir, "config.yaml")
    cam_ids = sorted(collect_camera_ids(rgbs_dir))
    with open(stage_config, "w") as f:
        f.write("data:\n")
        f.write("  pixel_source:\n")
        f.write(f"    cameras: {cam_ids}\n")

    return stage_dir


def collect_camera_ids(image_dir: str):
    cam_ids = set()
    for name in os.listdir(image_dir):
        match = re.match(r"\d+_(\d+)\.png$", name)
        if match:
            cam_ids.add(int(match.group(1)))
    return cam_ids


def main(args):
    log_dir1 = resolve_log_dir(args.log_dir1, prefer_parallel=False)
    log_dir2 = resolve_log_dir(args.log_dir2, prefer_parallel=True)

    if has_parallel_outputs(log_dir1):
        log_dir1 = stage_parallel_dir(log_dir1)
    if has_parallel_outputs(log_dir2):
        log_dir2 = stage_parallel_dir(log_dir2)

    logger.info(f"Comparing old single/model dir: {log_dir1}")
    logger.info(f"Comparing new view-parallel dir: {log_dir2}")

    compare_args = SimpleNamespace(
        log_dir1=log_dir1,
        log_dir2=log_dir2,
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        num_cams=args.num_cams,
    )
    compare.main(compare_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compare a single-model run with a view-parallel run")
    parser.add_argument("--log_dir1", type=str, required=True, help="Old single-model run directory")
    parser.add_argument("--log_dir2", type=str, required=True, help="New view-parallel run or project directory")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--num_cams", type=int, default=11, help="Number of cameras")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(parser.parse_args())
