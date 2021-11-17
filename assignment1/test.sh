#!/usr/bin/env bash

# python3.6 test.py $1 $2 attr.npy model/model1.h5 model/model2.h5 model/model3.h5 model/model4.h5

# ========================== Analysis ================================ #

# analysis
python main.py  $1 --test-only --model-name rsn11 --model-type resnet18  --save-dir checkpoints/  --test-output results/  --batch-size 32  --seed 211021  --world-size 1  --device-id 0


# ========================== Ensembled =============================== #

# .70298
# python main.py  AS1_data/test.csv --test-only --model-name ensemble8 --model-type vgg11 --ensemble  --model-list "('vgg1', 'vgg2', 'vgg3', 'vgg4', 'vgg5', 'vgg6')"  --save-dir checkpoints/  --test-output results/  --batch-size 32  --seed 211021  --world-size 1  --device-id 2

# .70047
# python main.py  AS1_data/test.csv --test-only --model-name ensemble9 --model-type vgg11 --ensemble  --model-list "('vgg1', 'vgg2', 'vgg3', 'vgg4', 'vgg6')"  --save-dir checkpoints/  --test-output results/  --batch-size 32  --seed 211021  --world-size 1  --device-id 2

# .70576
# python main.py  AS1_data/test.csv --test-only --model-name ensemble10 --model-type vgg11 --ensemble  --model-list "('rsn1', 'rsn2', 'rsn4', 'rsn5', 'rsn7', 'rsn9', 'vgg1', 'vgg2', 'vgg3', 'vgg4', 'vgg6')"  --save-dir checkpoints/  --test-output results/  --batch-size 32  --seed 211021  --world-size 1  --device-id 2

# .70855
# python main.py  AS1_data/test.csv --test-only --model-name ensemble11 --model-type vgg11 --ensemble  --model-list "('rsn1', 'rsn4', 'rsn5', 'rsn6', 'rsn7', 'rsn8', 'rsn9', 'rsn10', 'vgg1', 'vgg2', 'vgg3', 'vgg4', 'vgg5', 'vgg6')"  --save-dir checkpoints/  --test-output results/  --batch-size 32  --seed 211021  --world-size 1  --device-id 2

# .70716
# python main.py  AS1_data/test.csv --test-only --model-name ensemble12 --model-type vgg11 --ensemble  --model-list "('rsn1', 'rsn2', 'rsn3', 'rsn4', 'rsn5', 'rsn6', 'rsn7', 'rsn8', 'rsn9', 'rsn10', 'vgg1', 'vgg2', 'vgg3', 'vgg4', 'vgg5', 'vgg6')"  --save-dir checkpoints/  --test-output results/  --batch-size 32  --seed 211021  --world-size 1  --device-id 2

# .70994
# python main.py  AS1_data/test.csv --test-only --model-name ensemble13 --model-type vgg11 --ensemble  --model-list "('rsn1', 'rsn2', 'rsn5', 'rsn6', 'rsn7', 'rsn8', 'rsn9', 'vgg1', 'vgg2', 'vgg3', 'vgg4', 'vgg5', 'vgg6')"  --save-dir checkpoints/  --test-output results/  --batch-size 32  --seed 211021  --world-size 1  --device-id 2

# .71022
# python main.py  AS1_data/test.csv --test-only --model-name ensemble14 --model-type vgg11 --ensemble  --model-list "('rsn1', 'rsn2', 'rsn5', 'rsn6', 'rsn7', 'rsn8', 'rsn9', 'vgg1', 'vgg2', 'vgg3', 'vgg4', 'vgg5')"  --save-dir checkpoints/  --test-output results/  --batch-size 32  --seed 211021  --world-size 1  --device-id 2


# .70743
# python main.py  AS1_data/test.csv --test-only --model-name ensemble15 --model-type vgg11 --ensemble  --model-list "('rsn1', 'rsn2', 'rsn5', 'rsn6', 'rsn7', 'rsn8', 'vgg1', 'vgg3', 'vgg4', 'vgg5')"  --save-dir checkpoints/  --test-output results/  --batch-size 32  --seed 211021  --world-size 1  --device-id 2

# .70911
# python main.py  AS1_data/test.csv --test-only --model-name ensemble16 --model-type vgg11 --ensemble  --model-list "('rsn1', 'rsn2', 'rsn5', 'rsn6', 'rsn7', 'rsn8', 'vgg1', 'vgg2', 'vgg3', 'vgg4', 'vgg5')"  --save-dir checkpoints/  --test-output results/  --batch-size 32  --seed 211021  --world-size 1  --device-id 2

