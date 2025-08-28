# Chery-3DGS
Personal project codebase for chery reconstruction.

### update in 20250827

- 训练执行命令：

```
CUDA_VISIBLE_DEVICES=1 python tools/train.py     --config_file configs/omnire.yaml     --output_root output     --project omnire     --run_name waymo23_chery     dataset=waymo/1cams     data.scene_idx=0     data.start_timestep=0     data.end_timestep=-1
```

常更换参数：--config_file     --run_name   dataset   

- config_file修改内容
  - 注释model.SMPLNodes
  - 增添render.render_novel.traj_types 从 drivestudio/utils/camera.py 中选择
  - 根据是否使用lidar，注释model.Background.init
- --run_name
  - 代表输出日志文件夹名字
- dataset修改内容
  - 修改data.data_root为奇瑞数据路径
  - 根据是否使用lidar修改data.lidar_source.load_lidar data.lidar_source.only_use_top_lidar
  - 奇瑞数据需要修改data.pixel_source.undistort为False
  - dataset路径：drivestudio/configs/datasets/waymo
- 当前config_file各版本注释
  - 0827omnire_origin：初始版本waymo，具有SMPL属性
  - 0827omnire：无SMPL属性，无model.Background.init，纯视觉版本waymo
  - 0827omnire_chery：无SMPL属性，无model.Background.init，纯视觉版本奇瑞
  - 0827omnire_chery_lidar：无SMPL属性，结合雷达版本版本奇瑞
- 当前drivestudio/configs/datasets/waymo/1cams.yaml 各版本注释
  - 1cams, 3cams, 5cams 原始waymo的多相机版本
  - 0827_1cams_waymo：waymo的无雷达版本
  - 0827_1cams_chery：无雷达，单相机奇瑞
  - 0827_1cams_chery_lidar：有雷达，单相机奇瑞
  - 0827_5cams_chery：无雷达，5相机奇瑞
- 当前核心代码修改部分
  - drivestudio/tools/train.py 中156行 增加depth深度键值
  - drivestudio/tools/eval.py中220行 增加depth深度键值
  - drivestudio/datasets/waymo/waymo_sourceloader.py中429-454 中增加is_chery判断，chery数据初始化雷达14维数据
  - drivestudio/datasets/driving_dataset.py中290-296中增加self.lidar_source != None判断，随机初始化点云
  - drivestudio/utils/camera.py中增加多车道视觉变换渲染函数
### update in 20250828
- config_file修改内容
  - 0828omnire_chery_dist 视觉版本无变化
  - 0828omnire_chery_dist_60000 视觉版本训练改为60000次
  - 0828omnire_chery_dist_lidar 视觉+雷达版本 无变化
- dataset修改内容
  - 0828_1cams_chery_dist：pixel.source.undistort=False  downscale_when_loading=2   load_lidar=False
  - 0828_1cams_chery_dist_lidar: pixel.source.undistort=False  downscale_when_loading=2   load_lidar=True
  - 0828_1cams_chery_dist1_lidar：pixel.source.undistort=True  downscale_when_loading=1   load_lidar=True
  - 0828_1cams_chery_dist1：pixel.source.undistort=True  downscale_when_loading=2   load_lidar=False
  - ps：所有 downscale_when_loading=2 是因为修改了datasets/dataset_meta.py 的waymo 0的分辨率
