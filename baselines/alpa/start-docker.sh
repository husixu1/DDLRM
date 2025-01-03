#!/bin/bash

THISDIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
cd "$THISDIR" || exit 1

docker run -itd --rm \
    --name "alpa" \
    --gpus all --shm-size=10.24gb \
    -v "${THISDIR}":"/build/code" \
    husixu/alpa:v0.2
