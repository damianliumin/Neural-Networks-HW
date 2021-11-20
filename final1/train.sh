#!/usr/bin/env bash

# python main.py faces \
#     --model-name dcgan1 --model-type DCGAN \
#     --save-dir checkpoints/ --log-dir log/ \
#     --max-epoches 20  --batch-size 64 \
#     --seed 0 \
#     --optimizer Adam \
#     --lr 1e-4 \
#     --lr-scheduler constant \
#     --train-only \
#     --device-id 0

python main.py faces \
    --model-name wgan1 --model-type WGAN \
    --save-dir checkpoints/ --log-dir log/ \
    --max-epoches 100  --batch-size 64 \
    --seed 0 \
    --optimizer SGD \
    --lr 1e-4 \
    --lr-scheduler constant \
    --train-only \
    --device-id 0
