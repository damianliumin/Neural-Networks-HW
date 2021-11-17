# Created by LIU Min
# 191240030@smail.nju.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.LMNet import LMNet

class DeepEmotion(LMNet):
    def __init__(self, model_name, model_type, dropout_ratio=0, **kargs):
        super(DeepEmotion,self).__init__()

        self.name = model_name
        self.model_type = model_type
        self.dropout_ratio = dropout_ratio
        self.freeze_layer = None

        assert self.model_type == 'deepemotion'

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 10, 3, 1, 1),              # b x 10 x 48 x 48
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 10, 3, 1, 1),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2),                        # b x 10 x 24 x 24
            nn.ReLU(True),            
            # nn.Dropout(self.dropout_ratio),

            nn.Conv2d(10, 10, 3, 1, 1),             # b x 10 x 24 x 24
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 10, 3, 1, 1),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2),                        # b x 10 x 12 x 12
            nn.ReLU(True),            
            nn.Dropout(self.dropout_ratio),
        )

        self.fc = nn.Sequential(
            nn.Linear(1440, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),         # b x 10 x 42 x 42
            nn.MaxPool2d(2, stride=2),              # b x 10 x 21 x 21
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),        # b x 10 x 16 x 16
            nn.MaxPool2d(2, stride=2),              # b x 10 x 8 x 8
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )


        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))



    def stn(self, x_orig, x):
        xs = self.localization(x_orig)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,x):
        x_orig = x
        x = self.feature_extraction(x)
        x = self.stn(x_orig, x)
        x = x.reshape(-1, 1440)
        x = self.fc(x)
        return x


