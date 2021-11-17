# Created by LIU Min
# 191240030@smail.nju.edu.cn

import torch.nn as nn
import torchvision.models as models

from model.LMNet import LMNet

class Inception(LMNet):
    def __init__(self, model_name, model_type, dropout_ratio=0, pretrained=False, **kargs):
        super(Inception, self).__init__()

        self.name = model_name
        self.model_type = model_type
        self.dropout_ratio = dropout_ratio
        self.pretrained = pretrained
        self.freeze_layer = None


        if self.model_type in ('inception_v3'):
            self.inception = models.inception_v3(self.pretrained)
        else:
            raise RuntimeError(f'Unknown model type {self.model_type}')

        self.inception.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = self.inception(x)
        return x