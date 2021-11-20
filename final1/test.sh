#!/usr/bin/env bash

python main.py faces \
    --model-name dcgan1 --model-type DCGAN \
    --save-dir checkpoints/ --test-output results/ \
    --batch-size 64 \
    --seed 0 \
    --test-only \
    --device-id 0
