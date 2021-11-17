# Created by LIU Min
# 191240030@smail.nju.edu.cn

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as T


class emotionDataset(Dataset):
    def __init__(self, path, mode='train', args=None):
        """ pixel data normalized to [0, 1]
        """
        self.args = args
        self.mode = mode
        if not os.path.exists(path):
            raise RuntimeError(f'No such file or path: {path}')
            
        def str2numpy(strlist):
            strlist = list(strlist)
            rec = []
            for s in strlist:
                rec.append([int(i) for i in s.split()])
            return np.array(rec)


        data = pd.read_csv(path)
        if mode == 'test':
            self.data = torch.FloatTensor(str2numpy(data['pixels'])) / 255
            self.target = None
        else:
            # train : dev = 9 ： 1
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev' or mode == 'analysis':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            else:
                raise RuntimeError(f'Unknown data mode: {mode}')

            # data cache
            if (os.path.exists('./cache/train_data.pt') 
                and os.path.exists('./cache/train_target.pt')
                and mode == 'train' and args.cache):
                """ store target dur to shuffle """
                self.data = torch.load('./cache/train_data.pt')
                self.target = torch.load('./cache/train_target.pt')
                
            else:
                self.data = torch.FloatTensor(str2numpy(data['pixels'][indices])) / 255
                self.target = torch.LongTensor(data['emotion'][indices].values)
                if mode == 'train' and args.cache:
                    if not os.path.exists('./cache'):
                        os.mkdir('./cache')
                    torch.save(self.data, './cache/train_data.pt')
                    torch.save(self.target, './cache/train_target.pt')


        self.data = self.data.reshape(-1, 1, 48, 48)  # n x 1 x 48 x 48

        # data augmentation
        if self.mode == 'train':
            self.transform = T.Compose([
                T.RandomCrop(44),
                T.RandomHorizontalFlip(),
                T.Resize(48)
            ])
        elif self.mode in ('test', 'dev'):
            self.transform = T.Compose([
                T.TenCrop(44),
                T.Lambda(lambda crops: torch.stack([crop for crop in crops])),
                T.Resize(48)
            ])
        elif self.mode == 'analysis':
            self.transform = T.Compose([])

        # pretrained models input preprocessing
        if (args is not None 
            and (args.model_type.startswith('resnet') or args.model_type.startswith('vgg')
            or args.model_type.startswith('inception') or args.model_type.startswith('resnext'))) \
            or mode == 'analysis':
            self.data = self.data.repeat(1, 3, 1, 1)  # n x 3 x 48 x 48
            mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).repeat(1, 48, 48)
            std = torch.Tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).repeat(1, 48, 48)
            self.data = (self.data - mean) / std

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.transform(self.data[idx])
        else:
            return self.transform(self.data[idx]), self.target[idx]
    

def prep_dataloader(datapath, mode='train', batchsz=64, num_workers=0, args=None):
    dataset = emotionDataset(datapath, mode, args)
    dataloader = DataLoader(
        dataset, batchsz,
        shuffle=(mode=='train'),
        num_workers=num_workers
    )
    return dataloader


if __name__=='__main__':
    # test and debug
    dataloader = prep_dataloader('./AS1_data/', mode='test')
    for x in dataloader:
        print(x.shape, x.dtype)
        print(x.mean(), x.std())
        import matplotlib.pyplot as plt
        plt.imshow(x[3].reshape(48, 48) * 255, cmap='gray')
        plt.savefig('tmp.png')
        break


