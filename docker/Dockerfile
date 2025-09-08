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
RUN cp /opt/configs/.gitconfig /opt/configs/.vimrc /root/ && \
    cp /opt/configs/pip.conf /etc/ && \
    cp /etc/apt/sources.list /etc/apt/sources.list.backup && \
    cp /opt/configs/sources.list /etc/apt/ && \
    apt clean && apt update && \
    apt install -y software-properties-common && \
    add-apt-repository "ppa:ubuntugis/ppa" && \
    apt install -y aria2 wget clangd clang-format curl libgoogle-glog-dev libgflags-dev \
        vim sudo git build-essential cmake gdb net-tools libgl1 iputils-ping iproute2

RUN aria2c -x 16 -s 4 -m 3 -i /opt/downloads.txt --summary-interval=1

RUN bash /opt/downloads/Miniconda3-py39_25.7.0-2-Linux-x86_64.sh -b -u -p /opt/conda && \
    cp /opt/configs/.condarc /opt/conda && conda clean -i && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    mkdir -p /opt/local-conda-channel/linux-64 && \
    cp /opt/downloads/conda_pkgs/pytorch3d/* /opt/downloads/conda_pkgs/pytorch/* \
        /opt/local-conda-channel/linux-64/ && \
    conda install conda-build -y && conda index /opt/local-conda-channel

RUN /opt/retry.sh conda install pytorch=2.4.1 torchvision torchaudio pytorch-cuda=12.1 \
        -c file:///opt/local-conda-channel -c pytorch -c nvidia -y
RUN /opt/retry.sh conda install pytorch3d=0.7.8 -c file:///opt/local-conda-channel -c pytorch -y

ENV http_proxy=$MY_HTTP_PROXY https_proxy=$MY_HTTPS_PROXY

RUN pip install -U pip setuptools wheel && \
    pip install -r /opt/requirements.txt
RUN git clone https://github.com/NVlabs/SegFormer /opt/SegFormer && pip install /opt/SegFormer

ENV http_proxy= https_proxy=
