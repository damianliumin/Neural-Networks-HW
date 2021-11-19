#!/usr/bin/env bash

python main.py $1 $2 \
    --model-name dann3 --model-type DaNN \
    --save-dir checkpoints/ --test-output results/ \
    --batch-size 128 \
    --seed 0 \
    --test-only \
    --device-id 0
