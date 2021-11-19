#!/usr/bin/env bash

python main.py $1 $2 \
    --model-name dann4 --model-type DaNN \
    --save-dir checkpoints/ --log-dir log/ \
    --max-epoches 800  --batch-size 128 \
    --seed 2021 \
    --optimizer Adam \
    --lr 1e-3 \
    --lr-scheduler constant --weight-decay 0 \
    --train-only \
    --device-id 0
