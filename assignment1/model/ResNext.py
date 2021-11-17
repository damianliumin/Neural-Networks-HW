# Created by LIU Min
# 191240030@smail.nju.edu.cn

import torch.nn as nn
import torchvision.models as models

from model.LMNet import LMNet
 
class ResNext(LMNet):
    def __init__(self, model_name, model_type, dropout_ratio=0, pretrained=False, **kargs):
        super(ResNext, self).__init__()

        self.name = model_name
        self.model_type = model_type
        self.dropout_ratio = dropout_ratio
        self.pretrained = pretrained
        self.freeze_layer = None


        if self.model_type in ('resnext50_32x4d', 'resnext101_32x8d'):
            self.resnext = getattr(models, self.model_type)(self.pretrained)
        else:
            raise RuntimeError(f'Unknown model type {self.model_type}')

        # set fully-connected layers
        feature_size = self.resnext.fc.in_features
        if feature_size == 2048:
            self.resnext.fc = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(self.dropout_ratio),
                nn.Linear(512, self.num_classes)
            )
        elif feature_size == 512:
            self.resnext.fc = nn.Sequential(
                nn.Linear(512, self.num_classes)
            )
        else:
            raise RuntimeError('Pretrained model meets unknown feature size.')

        # add some dropout layers
        self.dp1 = nn.Dropout(self.dropout_ratio)
        self.dp2 = nn.Dropout(self.dropout_ratio)
        self.dp3 = nn.Dropout(self.dropout_ratio)
        self.dp4 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.resnext.conv1(x)
        x = self.resnext.bn1(x)
        x = self.resnext.relu(x)
        x = self.resnext.maxpool(x)

        x = self.resnext.layer1(x)
        x = self.dp1(x)
        x = self.resnext.layer2(x)
        x = self.dp2(x)
        x = self.resnext.layer3(x)
        x = self.dp3(x)
        x = self.resnext.layer4(x)
        x = self.dp4(x)

        x = self.resnext.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.resnext.fc(x)
        return x