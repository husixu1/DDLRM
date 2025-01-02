#!/bin/bash

THISDIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
cd "$THISDIR" || exit 1

# NOTE: --shm-size is important for preventing `ncclCommInitRank failed: unhandled system error`
docker run -itd --rm \
    --name "pipe" \
    --gpus all --shm-size=64gb \
    -v "datasets":"/root/.cache/hetu/datasets" \
    husixu/pipe:v0.4
