# Created by LIU Min
# 191240030@smail.nju.edu.cn

import cv2
import glob
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CrypkoDataset(Dataset):
    def __init__(self, fnames):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.fnames = fnames
        self.num_samples = len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img
    
    def __len__(self):
        return self.num_samples

    
def prep_dataloader(datapath, mode='train', batchsz=64, num_workers=0, args=None):
    fnames = glob.glob(os.path.join(datapath, '*'))
    dataset = CrypkoDataset(fnames)
    dataloader = DataLoader(
        dataset, batchsz,
        shuffle=(mode=='train'),
        num_workers=num_workers
    )
    return dataloader


if __name__=='__main__':
    # test and debug
    dataset = prep_dataloader('faces').dataset
    images = [(dataset[i] + 1) / 2 for i in range(16)]
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig('tmp.png', dpi=500)


