#!/bin/bash

function path_to_name() {
  local path="$1"
  local name="${path//\//-}"
  name="${name#-}"
  echo "${name}"
}

DOCKER_IMAGE_NAME="scene/pytorch-v0.1"
DOCKER_WS=/scene_reconstruction

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_NAME=$(path_to_name ${WS_DIR})

function fetch_docker_info() {
  echo $(docker ps -a --filter "name=${DOCKER_NAME}" --format "{{.ID}}|{{.Status}}")
}
