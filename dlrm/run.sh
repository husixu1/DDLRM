#!/bin/bash
case "$1" in
"kaggle")
    python dlrm_s_pytorch.py \
        --arch-sparse-feature-size=16 \
        --arch-mlp-bot="13-512-256-64-16" \
        --arch-mlp-top="512-256-1" \
        --data-generation=dataset \
        --data-set=kaggle \
        --raw-data-file=./input/train.txt \
        --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
        --loss-function=bce \
        --round-targets=True \
        --learning-rate=0.1 \
        --mini-batch-size=128 \
        --print-freq=1024 \
        --print-time \
        --test-mini-batch-size=16384 \
        --test-num-workers=16 \
        --inference-only
    ;;
"kaggle-pp")
    python dlrm_pp_inference.py \
        --arch-sparse-feature-size=16 \
        --arch-mlp-bot="13-512-256-64-16" \
        --arch-mlp-top="512-256-1" \
        --data-generation=dataset \
        --data-set=kaggle \
        --raw-data-file=./input/train.txt \
        --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
        --loss-function=bce \
        --round-targets=True \
        --learning-rate=0.1 \
        --mini-batch-size=128 \
        --print-freq=1024 \
        --print-time \
        --test-mini-batch-size=16384 \
        --test-num-workers=16 \
        --inference-only
    ;;
"random") python dlrm_s_pytorch.py --mini-batch-size=1 --data-size=1 --nepochs=100 --arch-interaction-op=dot --learning-rate=0.1 --inference-only ;;
"random-pp") python dlrm_pp_inference.py --mini-batch-size=1 --data-size=1 --nepochs=100 --arch-interaction-op=dot --learning-rate=0.1 --inference-only --use-gpu ;;
*) echo "Usage: $0 {kaggle|kaggle-pp|random|random-pp}" >&2 && exit 1 ;;
esac
