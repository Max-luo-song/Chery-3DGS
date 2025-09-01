FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV LANG="C.UTF-8"

ARG VIMRC="set nu\n\
set hlsearch\n\
set tabstop=4\n\
set shiftwidth=4\n\
set expandtab\n\
autocmd FileType cpp,c,h,hpp setlocal tabstop=2 shiftwidth=2 softtabstop=2\n\
autocmd FileType python setlocal tabstop=4 shiftwidth=4 softtabstop=4\n\
syntax on\n\
set autoindent\n\
set smartindent\n\
set cindent\n\
set showmode\n\
set showcmd\n\
set encoding=utf-8\n\
set fileencoding=utf-8\n\
"

ARG GIT_CONFIG="[core]\n\
    editor = vim\n\
"

ARG DEBIAN_FRONTEND=noninteractive

RUN cp /etc/apt/sources.list /etc/apt/sources.list.backup \
    && sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
    && apt clean \
    && apt update \
    && apt install -y software-properties-common \
    && add-apt-repository "ppa:ubuntugis/ppa" \
    && apt install -y clangd clang-format curl libgoogle-glog-dev libgflags-dev vim sudo git build-essential cmake gdb net-tools \
    && pip install yapf \
    && echo ${VIMRC} > /root/.vimrc \
    && echo ${GIT_CONFIG} > /root/.gitconfig

COPY docker_start_user.sh /tmp/
