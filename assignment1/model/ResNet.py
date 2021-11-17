# Created by LIU Min
# 191240030@smail.nju.edu.cn

import torch.nn as nn
import torchvision.models as models

from model.LMNet import LMNet
 
class ResNet(LMNet):
    def __init__(self, model_name, model_type, dropout_ratio=0, pretrained=False, **kargs):
        super(ResNet, self).__init__()

        self.name = model_name
        self.model_type = model_type
        self.dropout_ratio = dropout_ratio
        self.pretrained = pretrained
        self.freeze_layer = None

        if self.model_type in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
            self.resnet = getattr(models, self.model_type)(self.pretrained)
        else:
            raise RuntimeError(f'Unknown model type {self.model_type}')

        # set fully-connected layers
        feature_size = self.resnet.fc.in_features
        if feature_size == 2048:
            self.resnet.fc = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(self.dropout_ratio),
                nn.Linear(512, self.num_classes)
            )
        elif feature_size == 512:
            self.resnet.fc = nn.Sequential(
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
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.dp1(x)
        x = self.resnet.layer2(x)
        x = self.dp2(x)
        x = self.resnet.layer3(x)
        x = self.dp3(x)
        x = self.resnet.layer4(x)
        x = self.dp4(x)

        x = self.resnet.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.resnet.fc(x)
        return x

