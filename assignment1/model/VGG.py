# Created by LIU Min
# 191240030@smail.nju.edu.cn

import torch.nn as nn
import torchvision.models as models

from model.LMNet import LMNet
 
class VGG(LMNet):
    def __init__(self, model_name, model_type, dropout_ratio=0, pretrained=False, **kargs):
        super(VGG, self).__init__()

        self.name = model_name
        self.model_type = model_type
        self.dropout_ratio = dropout_ratio
        self.pretrained = pretrained
        self.freeze_layer = None


        if self.model_type in ('vgg11', 'vgg19'):
            self.vgg = getattr(models, self.model_type)(self.pretrained)
        else:
            raise RuntimeError(f'Unknown model type {self.model_type}')

        self.fc = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        x = self.vgg(x)
        x = self.fc(x)
        return x