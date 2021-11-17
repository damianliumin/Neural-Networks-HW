# Created by LIU Min
# 191240030@smail.nju.edu.cn

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from optimizer import LMOpt

class Trainer(object):
    def __init__(self, model, dataloader, device, args):
        self.model = model
        self.dataloader = dataloader
        self.epoch = 0
        self.optimizer = LMOpt(model, args)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.device = device

        self.train_loss = 0
        self.num_correct = 0
        self.num_data = len(dataloader.dataset)
        self.train_acc = 0

    def train_epoch(self):
        self.epoch += 1
        self.num_correct = 0
        self.train_loss = 0

        for x, y in tqdm(self.dataloader):
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 10) # clip gradient
            self.optimizer.step(self.epoch)
            self.optimizer.zero_grad()
            self.num_correct += np.sum((torch.argmax(F.softmax(self.model(x), dim=1), dim=1) == y).cpu().numpy())
            self.train_loss += len(x) * loss.item()
        
        self.train_acc = self.num_correct / self.num_data
        self.train_loss = self.train_loss / self.num_data

        # freeze layers for pretrained models
        self.model.freeze(self.epoch)

        return {
            'lr': self.optimizer.lr,
            'loss': self.train_loss,
            'acc': self.train_acc
        }

    @property
    def lr(self):
        return self.optimizer.lr



