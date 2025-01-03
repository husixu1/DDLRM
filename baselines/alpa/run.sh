#!/bin/bash
conda activate alpa

# see https://github.com/alpa-projects/alpa/issues/496
export NCCL_P2P_LEVEL=PIX # or PXB, see nvidia-smi topo -m
export NCCL_SHM_DISABLE=1

python ./test_mlp.py
