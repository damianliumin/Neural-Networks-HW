#!/usr/bin/env bash

# python3.6 train.py $1 model.h5

# ========================== Analysis ================================ #

# resnet without overfitting (for analysis)
python main.py  $1 --train-only --model-name rsn11 --model-type resnet18  --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 40  --batch-size 128  --dropout-ratio 0.3  --seed 31232  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.001  --world-size 1  --device-id 1  --early-stop 30 --cache


# ========================== Ensembled =============================== #

# resnet
# python main.py  AS1_data/train.csv --train-only --model-name rsn1 --model-type resnet18  --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 300  --batch-size 128  --dropout-ratio 0.3  --seed 31232  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.001  --world-size 1  --device-id 2  --early-stop 30 --cache

# python main.py  AS1_data/train.csv --train-only --model-name rsn2 --model-type resnet18  --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 300  --batch-size 128  --dropout-ratio 0.2  --seed 12341  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.001  --world-size 1  --device-id 2  --early-stop 30 --cache

# python main.py  AS1_data/train.csv --train-only --model-name rsn3 --model-type resnet18  --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 300  --batch-size 128  --dropout-ratio 0.1  --seed 12412  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.001  --world-size 1  --device-id 2  --early-stop 30 --cache

# python main.py  AS1_data/train.csv --train-only --model-name rsn4 --model-type resnet18  --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 300  --batch-size 128  --dropout-ratio 0.2  --seed 321  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.0003  --world-size 1  --device-id 2  --early-stop 30 --cache

# python main.py  AS1_data/train.csv --train-only --model-name rsn5 --model-type resnet18  --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 300  --batch-size 128  --dropout-ratio 0.3  --seed 131532  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.0001  --world-size 1  --device-id 2  --early-stop 30 --cache

# python main.py  AS1_data/train.csv --train-only --model-name rsn6 --model-type resnet34  --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 300  --batch-size 128  --dropout-ratio 0.3  --seed 523567  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.001  --world-size 1  --device-id 2  --early-stop 30 --cache

# python main.py  AS1_data/train.csv --train-only --model-name rsn7 --model-type resnet34  --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 300  --batch-size 128  --dropout-ratio 0.3  --seed 523567  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.0001  --world-size 1  --device-id 2  --early-stop 30 --cache

# python main.py  AS1_data/train.csv --train-only --model-name rsn8 --model-type resnet34  --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 300  --batch-size 128  --dropout-ratio 0.2  --seed 523567  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.0001  --world-size 1  --device-id 2  --early-stop 30 --cache

# python main.py  AS1_data/train.csv --train-only --model-name rsn9 --model-type resnet34  --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 300  --batch-size 128  --dropout-ratio 0.1  --seed 523567  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.0001  --world-size 1  --device-id 2  --early-stop 30 --cache

# python main.py  AS1_data/train.csv --train-only --model-name rsn10 --model-type resnet34  --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 300  --batch-size 128  --dropout-ratio 0.0  --seed 523567  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.0001  --world-size 1  --device-id 2  --early-stop 30 --cache


# vgg19
# python main.py  AS1_data/train.csv --train-only --model-name vgg1 --model-type vgg11 --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 200  --batch-size 64  --dropout-ratio 0.3  --seed 211021  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.001  --world-size 1  --device-id 3  --early-stop 30  --cache

# python main.py  AS1_data/train.csv --train-only --model-name vgg2 --model-type vgg11 --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 200  --batch-size 64  --dropout-ratio 0.3  --seed 412423  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.0001  --world-size 1  --device-id 3  --early-stop 30  --cache

# python main.py  AS1_data/train.csv --train-only --model-name vgg3 --model-type vgg19 --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 200  --batch-size 64  --dropout-ratio 0.3  --seed 12527  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.001  --world-size 1  --device-id 3  --early-stop 30  --cache

# python main.py  AS1_data/train.csv --train-only --model-name vgg4 --model-type vgg11 --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 200  --batch-size 64  --dropout-ratio 0.3  --seed 412423  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.0001  --world-size 1  --device-id 2  --early-stop 30  --cache

# python main.py  AS1_data/train.csv --train-only --model-name vgg5 --model-type vgg11 --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 200  --batch-size 64  --dropout-ratio 0.2  --seed 1241  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.0001  --world-size 1  --device-id 2  --early-stop 30  --cache

# python main.py  AS1_data/train.csv --train-only --model-name vgg6 --model-type vgg11 --pretrained  --save-dir checkpoints/  --log-dir log/  --test-output results/   --num-classes 7  --max-epoch 200  --batch-size 64  --dropout-ratio 0.2  --seed 1247  --warmup-init-lr 1e-4 --warmup-epoches 5  --lr 5e-3  --min-lr 1e-9 --lr-scheduler inverse_sqrt  --optimizer SGD --weight-decay 0.0001  --world-size 1  --device-id 2  --early-stop 30  --cache
