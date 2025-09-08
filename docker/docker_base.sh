#!/bin/bash

function path_to_name() {
  local path="$1"
  local name="${path//\//-}"
  name="${name#-}"
  echo "${name}"
}

DOCKER_IMAGE_NAME="scene/cuda12.1_py39_pyt241"
DOCKER_WS=/scene_reconstruction

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_NAME=$(path_to_name ${WS_DIR})

function docker_cmd() {
  local cmd="sudo docker"
  if docker ps >/dev/null 2>&1; then
    cmd="docker"
  fi
  echo ${cmd}
}

DOCKER_CMD=$(docker_cmd)

function fetch_docker_info() {
  echo $(${DOCKER_CMD} ps -a --filter "name=${DOCKER_NAME}" --format "{{.ID}}|{{.Status}}")
}
