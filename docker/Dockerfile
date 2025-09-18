FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG MY_HTTP_PROXY
ARG MY_HTTPS_PROXY

ENV LANG="C.UTF-8" \
    DEBIAN_FRONTEND=noninteractive \
    CONDA_AUTO_UPDATE_CONDA=false \
    PATH="${PATH}:/opt/conda/bin" \
    CONDA_ALWAYS_YES=true \
    CONDA_ACCEPT_LICENSES=yes

COPY copy/opt/ /opt/

SHELL ["/bin/bash", "-c"]

RUN cp /opt/configs/.gitconfig /opt/configs/.vimrc /root/ && \
    cp /opt/configs/pip.conf /etc/ && \
    cp /etc/apt/sources.list /etc/apt/sources.list.backup && \
    cp /opt/configs/sources.list /etc/apt/ && \
    apt clean && apt update && \
    apt install -y software-properties-common && \
    add-apt-repository "ppa:ubuntugis/ppa" && \
    apt install -y aria2 wget clangd clang-format curl libgoogle-glog-dev libgflags-dev \
        vim sudo git build-essential cmake gdb net-tools libgl1 iputils-ping iproute2

ENV http_proxy=$MY_HTTP_PROXY https_proxy=$MY_HTTPS_PROXY

RUN git clone https://github.com/NVlabs/SegFormer.git /opt/SegFormer && \
    mkdir -p /opt/SegFormer/pretrained && \
    git clone https://github.com/open-mmlab/mmsegmentation.git /opt/mmsegmentation && \
    cd /opt/mmsegmentation && git checkout v1.2.2 && mkdir -p /opt/mmsegmentation/checkpoints

RUN aria2c -x 8 -s 8 -m 3 -i /opt/downloads.txt --summary-interval=1

RUN bash /opt/downloads/Miniconda3-py39_25.7.0-2-Linux-x86_64.sh -b -u -p /opt/conda && \
    cp /opt/configs/.condarc /opt/conda && conda clean -i && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda install conda-build pysocks -y && conda index /opt/local-conda-ch/main && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /etc/profile

RUN conda create -y -n main python=3.9 && source /opt/conda/etc/profile.d/conda.sh && \
    conda activate main && \
    conda install pytorch=2.3.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y && \
    conda install pytorch3d=0.7.8 -c file:///opt/local-conda-ch/main -y && \
    pip install -r /opt/requirements.txt

RUN conda create -y -n segformer python=3.8 && source /opt/conda/etc/profile.d/conda.sh && \
    conda activate segformer && \
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install timm==0.3.2 pylint debugpy opencv-python-headless attrs ipython tqdm imageio \
        scikit-image omegaconf && pip install mmcv-full==1.2.7 && \
        cd /opt/SegFormer && pip install .

RUN conda create -y -n mmseg2 python=3.9 && source /opt/conda/etc/profile.d/conda.sh && \
    conda activate mmseg2 && \
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install timm pylint debugpy opencv-python-headless attrs ipython tqdm imageio scikit-image omegaconf \
        ftfy regex numpy==1.26.4 opencv-python==4.11.0.86  && \
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html && \
    cd /opt/mmsegmentation && pip install .

ENV http_proxy=http://172.26.255.17:3128 \
    https_proxy=http://172.26.255.17:3128 \
    ftp_proxy=http://172.26.255.17:3128
