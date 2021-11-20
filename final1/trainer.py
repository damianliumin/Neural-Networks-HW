# Created by LIU Min
# 191240030@smail.nju.edu.cn

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.activation import SELU
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from optimizer import LMOpt


class Trainer(object):
    def __init__(self, model, dataloader, device, args):
        self.model = model
        self.discriminator = self.model.discriminator
        self.generator = self.model.generator
        self.model_type = args.model_type
        self.dataloader = dataloader
        self.epoch = 0
        self.steps = 0
        self.max_epoches = args.max_epoches
        
        if self.model_type == 'DCGAN':
            self.opt_D = optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(0.5,0.999))
            self.opt_G = optim.Adam(self.generator.parameters(), lr=args.lr, betas=(0.5,0.999))
        elif self.model_type == 'WGAN':
            self.opt_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=args.lr)
            self.opt_G = torch.optim.RMSprop(self.generator.parameters(), lr=args.lr)

        self.criterion = nn.BCELoss()

        self.n_critic = 1 if self.model_type == 'DCGAN' else 5

        self.device = device
        self.num_data = len(dataloader.dataset)

    def train_epoch(self):
        self.epoch += 1
        self.discriminator.train()
        self.generator.train()

        running_D_loss, running_G_loss = 0.0, 0.0

        for i, imgs in enumerate(self.dataloader):
            imgs = imgs.to(self.device)
            bs = imgs.size(0)

            #  Train Discriminator
            z = Variable(torch.randn(bs, 100)).to(self.device)
            r_imgs = Variable(imgs).to(self.device)
            f_imgs = self.model.generator(z)

            if self.model_type == 'DCGAN':
                # Label
                r_label = torch.ones((bs)).to(self.device)
                f_label = torch.zeros((bs)).to(self.device)

                # Model forwarding
                r_logit = self.discriminator(r_imgs.detach())
                f_logit = self.discriminator(f_imgs.detach())
                
                # Compute the loss for the discriminator.
                r_loss = self.criterion(r_logit, r_label)
                f_loss = self.criterion(f_logit, f_label)
                loss_D = (r_loss + f_loss) / 2
            elif self.model_type == 'WGAN':
                loss_D = -torch.mean(self.discriminator(r_imgs)) + torch.mean(self.discriminator(f_imgs))

            # Model backwarding
            self.discriminator.zero_grad()
            loss_D.backward()
            running_D_loss += loss_D.item()

            # Update the discriminator.
            self.opt_D.step()

            """ Medium: Clip weights of discriminator. """
            if self.model_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            #  Train Generator
            if self.steps % self.n_critic == 0:
                # Generate some fake images.
                z = Variable(torch.randn(bs, 100)).cuda()
                f_imgs = self.generator(z)

                # Model forwarding
                f_logit = self.discriminator(f_imgs)
                
                """ Medium: Use WGAN Loss"""
                if self.model_type == 'DCGAN':
                    # Compute the loss for the generator.
                    loss_G = self.criterion(f_logit, r_label)
                elif self.model_type == 'WGAN':
                    # WGAN Loss
                    loss_G = -torch.mean(self.discriminator(f_imgs))

                # Model backwarding
                self.generator.zero_grad()
                loss_G.backward()
                running_G_loss += loss_G.item()

                # Update the generator.
                self.opt_G.step()

            self.steps += 1

        return {
            'G loss': running_D_loss / (i+1), 
            'D loss': running_G_loss / (i+1), 
        }

    @property
    def lr(self):
        return 1e-3

