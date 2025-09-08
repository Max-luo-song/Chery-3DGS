# 准备奇瑞数据集
基于 Waymo 数据集版本修改

## 1. 设置数据集目录
```shell
# Create the data directory or create a symbolic link to the data directory
mkdir -p ./data/chery
mkdir -p ./data/chery/processed 

ln -s $PATH_TO_CHERY ./data/chery/raw
```

## 2. 数据预处理
#### 运行预处理脚本
对特定场景预处理：
```shell
export PYTHONPATH=\path\to\project

python datasets/preprocess.py \
    --data_root data/chery/raw/ \
    --target_dir data/chery/processed \
    --dataset chery \
    --split training \
    --scene_ids clip_1746752396800 \
    --workers 8 \
    --process_keys images lidar calib pose dynamic_masks objects
```
Alternatively, preprocess a batch of scenes by providing the split file:
```shell
export PYTHONPATH=\path\to\project

python datasets/preprocess.py \
    --data_root data/chery/raw/ \
    --target_dir data/chery/processed \
    --dataset chery \
    --split training \
    --split_file data/chery_example_scenes.txt \
    --workers 8 \
    --process_keys images lidar calib pose dynamic_masks objects
```
The extracted data will be stored in the `data/chery/processed` directory.

## 3. Extract Masks

To generate:

- **sky masks (required)** 
- fine dynamic masks (optional)

Follow these steps:

#### Install `SegFormer` (Skip if already installed)

:warning: SegFormer relies on `mmcv-full=1.2.7`, which relies on `pytorch=1.8` (pytorch<1.9). Hence, a seperate conda env is required.

```shell
#-- Set conda env
conda create -n segformer python=3.8
conda activate segformer
# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#-- Install mmcv-full
pip install timm==0.3.2 pylint debugpy opencv-python-headless attrs ipython tqdm imageio scikit-image omegaconf
pip install mmcv-full==1.2.7 --no-cache-dir

#-- Clone and install segformer
git clone https://github.com/NVlabs/SegFormer
cd SegFormer
pip install .
```

Download the pretrained model `segformer.b5.1024x1024.city.160k.pth` from the google_drive / one_drive links in https://github.com/NVlabs/SegFormer#evaluation .

Remember the location where you download into, and pass it to the script in the next step with `--checkpoint` .


#### Run Mask Extraction Script

```shell
conda activate segformer
segformer_path=/pathtosegformer

python datasets/tools/extract_masks.py \
    --data_root data/chery/processed/training \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --split_file data/chery_example_scenes.txt \
    --process_dynamic_mask
```
Replace `/pathtosegformer` with the actual path to your Segformer installation.

Note: The `--process_dynamic_mask` flag is included to process fine dynamic masks along with sky masks.

This process will extract the required masks from your processed data.

## 4. Human Body Pose Processing

#### Prerequisites
To utilize the SMPL-Gaussian to model pedestrians, please first download the SMPL models.

1. Download SMPL v1.1 (`SMPL_python_v.1.1.0.zip`) from the [SMPL official website](https://smpl.is.tue.mpg.de/download.php)
2. Move `SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to `PROJECT_ROOT/smpl_models/SMPL_NEUTRAL.pkl`

SMPL-Nodes (SMPL-Gaussian Representation) requires Human Body Pose Sequences of pedestrians. We've developed a human body pose processing pipeline for in-the-wild driving data to generate this information. There are two ways to obtain these data:

####  Run the Extraction Pipeline
To process human body poses, follow the instructions in our [Human Pose Processing Guide](./HumanPose.md).

## 5. Data Structure
After completing all preprocessing steps, the project files should be organized according to the following structure:
```shell
ProjectPath/data/
  └── chery/
    ├── raw/
    │    ├── clip_1746752396800
    │    └── ...
    └── processed/
         └──training/
                ├── 001/
                │  ├──images/             # Images: {timestep:03d}_{cam_id}.jpg
                │  ├──lidar/              # LiDAR data: {timestep:03d}.bin
                │  ├──lidar_pose/         # LiDAR poses: {timestep:03d}.txt
                │  ├──extrinsics/         # Camera extrinsics: {cam_id}.txt
                │  ├──intrinsics/         # Camera intrinsics: {cam_id}.txt
                │  ├──sky_masks/          # Sky masks: {timestep:03d}_{cam_id}.png
                │  ├──dynamic_masks/      # Coarse dynamic masks: category/{timestep:03d}_{cam_id}.png
                │  ├──fine_dynamic_masks/ # (暂时没有) Fine dynamic masks: category/{timestep:03d}_{cam_id}.png
                │  ├──instances/          # Instances' bounding boxes information
                │  └──humanpose/          # （暂时没有）Preprocessed human body pose: smpl.pkl
                ├── 002/
                ├── ...
```