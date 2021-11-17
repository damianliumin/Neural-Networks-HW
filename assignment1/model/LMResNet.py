# Created by LIU Min
# 191240030@smail.nju.edu.cn

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from model.LMNet import LMNet

class LMResNet(LMNet):
    def __init__(self, model_name, model_type, dropout_ratio=0, **kargs):
        super(LMResNet, self).__init__()

        self.name = model_name
        self.model_type = model_type
        self.dropout_ratio = dropout_ratio
        self.freeze_layer = None

        assert self.model_type == 'lmresnet'

        self.layer1 = nn.Sequential(
            self._make_layer(1, 16, 2, 1),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_ratio)
        )
        self.layer2 = nn.Sequential(
            self._make_layer(16, 32, 3, 1),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_ratio)
        )
        self.layer3 = nn.Sequential(
            self._make_layer(32, 64, 4, 1),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_ratio),
        )
        self.layer4 = nn.Sequential(
            self._make_layer(64, 128, 3, 1),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_ratio),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )
        
        self._init_params()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)
                if m.bias is not None:
                    init.constant(m.bias, 0)


class BasicBlock(nn.Module):
    """ ResNet """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride > 1 or self.in_channels != self.out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)

    def forward(self, x):
        residual = x
        x = self.layers(x)
        if self.stride > 1 or self.in_channels != self.out_channels:
            residual = self.downsample(residual)
        return F.relu(x + residual)