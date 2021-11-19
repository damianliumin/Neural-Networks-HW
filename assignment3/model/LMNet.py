# Created by LIU Min
# 191240030@smail.nju.edu.cn

import torch.nn as nn


class LMNet(nn.Module):
    """ model wrapper for my implementations """
    def __init__(self):
        super(LMNet, self).__init__()
        self.num_classes = 7

    def forward(self, x):
        raise RuntimeError("function: foward has not been implemented")

    def freeze(self, epoch):
        if not self.pretrained or self.freeze_layer is None:
            return
        if epoch in self.freeze_layers:
            layers = self.freeze_layers[epoch]
        else:
            return
        for name, p in self.pretrained_model.named_parameters():
                for layer in layers:
                    if name.startswith(layer):
                        p.requires_grad = False
                        break
        print(f'Freeze layers: {layers}.')
