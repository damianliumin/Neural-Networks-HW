# Created by LIU Min
# 191240030@smail.nju.edu.cn

import torch
import torch.nn as nn
from model.LMNet import LMNet

class DCGAN(LMNet):
    def __init__(self, model_name, model_type, dropout_ratio=0, **kargs):
        super(DCGAN,self).__init__()

        self.name = model_name
        self.model_type = model_type
        self.dropout_ratio = dropout_ratio
        self.freeze_layer = None

        assert self.model_type in ('DCGAN', 'WGAN')

        self.generator = Generator(in_dim=100)
        self.discriminator = Discriminator(in_dim=3, model_type=model_type)

    def forward(self, x):
        x = self.generator(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):
    """
    Input shape: (N, 3, 64, 64)
    Output shape: (N, )
    """
    def __init__(self, in_dim, dim=64, model_type='DCGAN'):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )
            
        """ Medium: Remove the last sigmoid layer for WGAN. """
        if model_type == 'DCGAN':
            self.ls = nn.Sequential(
                nn.Conv2d(in_dim, dim, 5, 2, 2), 
                nn.LeakyReLU(0.2),
                conv_bn_lrelu(dim, dim * 2),
                conv_bn_lrelu(dim * 2, dim * 4),
                conv_bn_lrelu(dim * 4, dim * 8),
                nn.Conv2d(dim * 8, 1, 4),
                nn.Sigmoid(), 
            )
        else:
            self.ls = nn.Sequential(
                nn.Conv2d(in_dim, dim, 5, 2, 2), 
                nn.LeakyReLU(0.2),
                conv_bn_lrelu(dim, dim * 2),
                conv_bn_lrelu(dim * 2, dim * 4),
                conv_bn_lrelu(dim * 4, dim * 8),
                nn.Conv2d(dim * 8, 1, 4),
            )
        self.apply(weights_init)
        
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y



