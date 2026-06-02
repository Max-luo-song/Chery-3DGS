# Qcraft View-Parallel Training And Rendering

本文档记录当前项目中针对 Qcraft 11 视角数据新增的两卡视角并行训练、评估和结果合并方案。

## 目标

当前改造实现的是“视角并行”，不是把 batch 拆到多张 GPU 上：

- GPU 0 训练并推理 Group 0 视角。
- GPU 1 训练并推理 Group 1 视角。
- 两个 GPU 分别训练两个独立模型权重。
- 最终评估结果在顶层目录统一汇总，保持接近原单模型输出结构。

默认视角划分：

```text
Group 0: [0, 2, 5, 6, 7, 12]
Group 1: [1, 3, 9, 10, 11]
```

完整视角顺序：

```text
[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12]
```

## 修改文件

- `configs/omnire_extended_cam_lidar.yaml`
  - 新增顶层 `view_parallel` 默认配置。

- `utils/view_parallel_utils.py`
  - 新增 view group 解析、`data.pixel_source.cameras` 覆盖、输出目录追加和日志打印逻辑。

- `tools/train.py`
  - 新增 view-parallel 训练参数。
  - 启用 view-parallel 时，训练输出自动进入 `group0/` 或 `group1/`。
  - 启用 view-parallel 时，默认跳过训练结束时的分组最终评估，避免每个 group 单独生成训练后视频。

- `tools/eval.py`
  - 新增 view-parallel 推理参数。
  - 支持按 group 加载对应 checkpoint 和视角子集。
  - 新增跳过分组 mp4 组装的参数，分组评估只保留合并所需的临时单相机图和指标。

- `tools/merge_view_parallel_render.py`
  - 新增渲染结果合并工具。
  - 从 `group0/videos_eval` 和 `group1/videos_eval` 读取单相机图片，按完整 11 视角 layout 合并到顶层 `videos/`。

- `tools/merge_view_parallel_metrics.py`
  - 新增指标合并工具。
  - 从 `group0/metrics_eval` 和 `group1/metrics_eval` 读取指标，按视角数量加权合并到顶层 `metrics_eval/`。

- `scripts/qcraft/train_parallel.sh`
  - 新增统一入口：两卡训练、两卡评估、视频合并、指标合并和临时分组视频目录清理。

## 新增配置

在 `configs/omnire_extended_cam_lidar.yaml` 中新增：

```yaml
view_parallel:
  enabled: false
  num_groups: 2
  group_id: null
  gpu_id: null
  groups:
    group0: [0, 2, 5, 6, 7, 12]
    group1: [1, 3, 9, 10, 11]
```

默认 `enabled: false`，因此原单卡全视角训练逻辑不受影响。

## 新增命令行参数

`tools/train.py` 和 `tools/eval.py` 均新增：

```text
--view_parallel
--num_view_groups 2
--view_group_id 0
--view_ids 0,2,5,6,7,12
--gpu_id 0
```

`tools/eval.py` 额外新增：

```text
--eval_output_dir
--skip_video_assembly
```

`tools/train.py` 额外新增：

```text
--run_final_eval
```

默认 view-parallel 训练不执行训练结束时的分组最终评估；如确实需要保留旧行为，可以显式传入 `--run_final_eval`。

## 两卡并行训练、评估和合并

默认一键命令：

```bash
bash scripts/qcraft/train_parallel.sh
```

默认配置：

```text
scene_idx=20250818_152739_Q3720_100_130_part01
config_file=configs/omnire_extended_cam_lidar.yaml
dataset_config=qcraft/11cams_lidar
group0_gpu=0
group1_gpu=1
group0_views=0,2,5,6,7,12
group1_views=1,3,9,10,11
```

也可以通过环境变量覆盖：

```bash
date_str=20260506 bash scripts/qcraft/train_parallel.sh
```

或指定输出根目录：

```bash
output_root=/inspire/hdd/global_user/guoluosong-253108120129/chery/new_traj/scene_reconstruction-main/output \
date_str=20260506 \
bash scripts/qcraft/train_parallel.sh
```

## 输出结构

默认输出根目录：

```text
output/
```

默认 run 目录：

```text
output/qcraft_20250818_152739_Q3720_100_130_part01/<date>_lidar+cam0_1_2_3_5_6_7_9_10_11_12baseline/
```

训练和评估后的结构：

```text
<run_name>/
  group0/
    checkpoint_final.pth
    config.yaml
    images/
    videos/
    metrics/
    metrics_eval/

  group1/
    checkpoint_final.pth
    config.yaml
    images/
    videos/
    metrics/
    metrics_eval/

  videos/
    rgbs/
    rgbs_layout/
    rgbs_layout.mp4
    depths/
    depths_layout/
    depths_layout.mp4
    ...

  metrics_eval/
    images_full_view_parallel_merged.json
    images_test_view_parallel_merged.json
```

用户最终主要查看顶层：

```text
<run_name>/videos/
<run_name>/metrics_eval/
```

`group0/` 和 `group1/` 保留为中间模型、分组渲染和分组指标目录。

默认 `cleanup_group_eval=1`，合并完成后会删除：

```text
group0/videos_eval/
group1/videos_eval/
```

因此最终不会保留每个 group 单独的评估视频目录。若调试时需要保留临时单相机评估图片，可以运行：

```bash
cleanup_group_eval=0 bash scripts/qcraft/train_parallel.sh
```

## 指标合并方式

两个 group 会各自评估自己的视角子集：

```text
group0: [0, 2, 5, 6, 7, 12]
group1: [1, 3, 9, 10, 11]
```

最终顶层指标按视角数量加权：

```text
group0 权重: 6 / 11
group1 权重: 5 / 11
```

合并输出：

```text
metrics_eval/images_full_view_parallel_merged.json
metrics_eval/images_test_view_parallel_merged.json
```

## 恢复原单卡全视角训练

原脚本仍可直接使用：

```bash
bash scripts/qcraft/train.sh
```

或者直接运行 `tools/train.py` 时不传 `--view_parallel`，并保持：

```text
data.pixel_source.cameras=[0,1,2,3,5,6,7,9,10,11,12]
```

## 当前局限

- 两个 group 是两个独立模型，不共享 Gaussian、动态节点或优化状态。
- 最终视频是后处理按相机 id 合并，不是单个模型一次性渲染 11 个视角。
- 指标合并使用视角数量加权，假设每个视角的帧数一致。
- 如果某些动态对象只在另一个 group 中可见，当前 group 模型不会获得另一组视角的监督。
- 顶层 `videos/` 和 `metrics_eval/` 是统一结果；`group0/`、`group1/` 仍会保留各自的中间输出。

## 验证

已完成静态检查：

```bash
python -m py_compile utils/view_parallel_utils.py tools/train.py tools/eval.py tools/merge_view_parallel_render.py tools/merge_view_parallel_metrics.py
bash -n scripts/qcraft/train_parallel.sh
```

未实际启动完整训练或评估。
