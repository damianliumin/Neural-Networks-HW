# Created by LIU Min
# 191240030@smail.nju.edu.cn

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

    
def prep_dataloader(datapath, mode='train', batchsz=64, num_workers=0, args=None):
    if 'train' in datapath:  # source
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Lambda(lambda x: cv2.Canny(np.array(x), 250, 300)),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    elif 'test' in datapath:  # target
        if mode == 'train':
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((32, 32)),
                transforms.RandomAffine(10 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
    else:
        assert False
    dataset = ImageFolder(datapath, transform=transform)
    dataloader = DataLoader(
        dataset, batchsz,
        shuffle=(mode=='train'),
        num_workers=num_workers
    )
    return dataloader


if __name__=='__main__':
    # test and debug
    dataloader = prep_dataloader('./AS2_data/train_data', mode='train', batchsz=1)
    for x, y in dataloader:
        titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
        print(x, titles[y])
        print(x.shape, x.dtype)
        print(x.mean(), x.std())
        import matplotlib.pyplot as plt
        plt.imshow(x[0, 0], cmap='gray')
        plt.show()
        break


