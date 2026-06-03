# BDGS 插入说明

## 1. 参考范围与适配结论

本次移植参考 `/inspire/hdd/global_user/guoluosong-253108120129/project/bilateral-drivinginsertbyyx` 中基于 OmniRe 的 BDGS（Multi-Scale Bilateral Grids）接入方式。参考工程工作区同时存在 `insertbyyx` 路面插件修改，因此移植时仅采用 BDGS 的外观校正链路，不覆盖本项目已有的 `RoadNodes`、LiDAR 深度监督、动态区域掩码及 Chery/Qcraft 数据扩展。

本项目与参考工程的关键结构差异是：本项目将通用 Affine 处理和损失集中在 `models/trainers/base.py`，而场景图前向仍由 `models/trainers/scene_graph.py` 维护路面相关输出。因此 BDGS 逻辑接入基类，场景图仅补充循环一致性损失所需的原始渲染颜色。

## 2. 新增文件

### `bilateral/__init__.py`

导出双边网格公共接口。

### `bilateral/lib_bilagrid.py`

新增 BDGS 运行核心：

- `BilateralGrid`：每张训练图像对应一个可学习的三维双边网格，参数初始化为恒等 `3 x 4` 颜色变换。
- `slice_grid`：按像素位置及 RGB 灰度 guidance 对网格进行三线性采样，输出逐像素 Affine 矩阵。
- `total_variation_loss`：对网格三个空间/引导维度施加 TV 正则。
- 测试帧回退策略：测试时间没有独立优化过的图像网格时，使用全部网格参数的均值采样；该行为与原有 `AffineTransform` 对测试帧使用均值外观编码的语义一致，并避免参考工程依赖未在本项目设置的 `training_indices_for_test`。

### `configs/omnire_ms_bilateral_lidar.yaml`

由 `configs/omnire_lidar.yaml` 派生的 BDGS 配置，保留原有 LiDAR/路面配置，只替换外观模块及其正则和优化器参数。

### `configs/omnire_ms_bilateral_extended_cam_lidar.yaml`

由 `configs/omnire_extended_cam_lidar.yaml` 派生的 BDGS 配置，适用于当前 Chery/Qcraft 多相机训练路径。

## 3. 修改文件

### `models/modules.py`

新增 `MultiScaleBilateralAffineTransform`：

- 使用三个尺度的网格 `[[2, 2, 1], [4, 4, 2], [8, 8, 4]]`。
- 三个尺度依次对渲染 RGB 执行逐像素 Affine 校正。
- guidance 采样按尺度降分辨率后再上采样 Affine 矩阵，控制显存和计算量。
- 暴露 `tv_loss()` 和 `inverse_loss()`，对应 BDGS 的网格平滑正则及可选循环一致性正则。
- 以 `Affine#grid0`、`Affine#grid1`、`Affine#grid2` 注册优化参数，兼容现有 Trainer 优化器初始化方式。

### `models/trainers/base.py`

- `affine_transformation()` 增加 `MultiScaleBilateralAffineTransform` 分支；未选择 BDGS 配置时仍执行原 `AffineTransform` 路径。
- 前向结果增加 `outputs["original_rgb"]`，保存外观校正前的 Gaussian 与天空混合颜色。
- Affine 正则增加 BDGS 分支：`w * TV + w1 * inverse_loss`。循环项基于完整有效图像区域计算，不受本项目单独的非路面 RGB 损失掩码影响。

### `models/trainers/scene_graph.py`

场景图前向增加 `outputs["original_rgb"]` 后再调用 Affine 校正，以支持 BDGS 循环一致性项；路面渲染、冻结逻辑及现有输出字段保持不变。

## 4. BDGS 配置参数

新增配置中的主要参数如下：

```yaml
trainer:
  losses:
    affine:
      w: 0.01
      w1: 0.0
model:
  Affine:
    type: models.modules.MultiScaleBilateralAffineTransform
    params:
      grid: [[2, 2, 1], [4, 4, 2], [8, 8, 4]]
    optim:
      grid0: {lr: 6.0e-4, lr_final: 3.0e-5, warmup_steps: 1000, lr_pre_warmup: 0}
      grid1: {lr: 6.0e-4, lr_final: 3.0e-5, warmup_steps: 1000, lr_pre_warmup: 0}
      grid2: {lr: 6.0e-4, lr_final: 3.0e-5, warmup_steps: 1000, lr_pre_warmup: 0}
```

`w1` 默认沿用参考 BDGS 配置设为 `0.0`；需要实验循环一致性约束时可通过命令行覆盖，例如 `trainer.losses.affine.w1=0.01`。

## 5. 使用方式

多相机 LiDAR 场景使用：

```bash
python tools/train.py \
  --config_file configs/omnire_ms_bilateral_extended_cam_lidar.yaml \
  --output_root output \
  --project <project_name> \
  --run_name <run_name> \
  dataset=<dataset_config> \
  data.scene_idx=<scene_idx>
```

基础 LiDAR OmniRe 场景将配置替换为：

```bash
--config_file configs/omnire_ms_bilateral_lidar.yaml
```

现有脚本也可直接覆盖配置路径，例如：

```bash
config_file=configs/omnire_ms_bilateral_extended_cam_lidar.yaml bash scripts/qcraft/train_view_parallel.sh
```

## 6. 验证记录

已在项目可用环境 `/opt/conda/envs/drivestudio/bin/python` 执行：

- `/opt/conda/envs/drivestudio/bin/python -m py_compile bilateral/__init__.py bilateral/lib_bilagrid.py models/modules.py models/trainers/base.py models/trainers/scene_graph.py`：通过。
- `OmegaConf.load()` 加载两份新增 BDGS 配置，并检查模块类型、三个网格尺度和 Affine 正则参数：通过。
- `BilateralGrid` 小张量烟测：输出矩阵尺寸正确，恒等初始化保持 RGB 不变，初始 TV 为零，测试帧均值回退可运行：通过。
- `MultiScaleBilateralAffineTransform` 小张量烟测：三个尺度输出尺寸正确，级联恒等映射、TV 正则和逆变换损失均符合预期：通过。

未执行完整场景训练或 GPU 渲染评估；这需要实际数据路径、训练时长及可用 GPU。
