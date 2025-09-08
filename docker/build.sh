#!/bin/bash

source ./docker_base.sh

my_http_proxy=""
my_https_proxy=""

while [ $# -gt 0 ]; do
  case "$1" in
    --http_proxy=*)
      my_http_proxy="${1#*=}"; shift ;;
    --https_proxy=*)
      my_https_proxy="${1#*=}"; shift ;;
    *) shift ;;
  esac
done

echo "http_proxy: ${my_http_proxy}"
echo "https_proxy: ${my_https_proxy}"

cp ../requirements.txt ./copy/opt/

${DOCKER_CMD} build \
  --build-arg MY_HTTP_PROXY=${my_http_proxy} \
  --build-arg MY_HTTPS_PROXY=${my_https_proxy} \
  -t scene/cuda12.1_py39_pyt241 .

rm ./copy/opt/requirements.txt
