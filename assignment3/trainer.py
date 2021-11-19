# Created by LIU Min
# 191240030@smail.nju.edu.cn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from optimizer import LMOpt


def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

class Trainer(object):
    def __init__(self, model, source_dataloader, target_dataloader, device, args):
        self.model = model
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.epoch = 0
        self.max_epoches = args.max_epoches
        self.optimizer_F = optim.Adam(model.feature_extractor.parameters(), lr=1e-4)
        self.optimizer_C = optim.Adam(model.label_predictor.parameters(), lr=1e-3)
        self.optimizer_D = optim.Adam(model.domain_classifier.parameters(), lr=1e-3)
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()
        self.device = device
        self.lamb = args.lamb
        self.num_data = 0

    def train_epoch(self):
        self.epoch += 1
        lamb = 0.1 * get_lambda(self.epoch, self.max_epoches)

        running_D_loss, running_F_loss = 0.0, 0.0
        total_hit, total_num = 0.0, 0.0

        for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(self.source_dataloader, self.target_dataloader)):

            source_data = source_data.to(self.device)
            source_label = source_label.to(self.device)
            target_data = target_data.to(self.device)
            
            mixed_data = torch.cat([source_data, target_data], dim=0)
            domain_label1 = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(self.device)
            domain_label1[:source_data.shape[0]] = 1
            # Step 1: Train Domain Classifier
            feature1 = self.model.feature_extractor(mixed_data)
            domain_logits = self.model.domain_classifier(feature1.detach())
            loss = self.domain_criterion(domain_logits, domain_label1)
            running_D_loss+= loss.item()
            loss.backward()
            self.optimizer_D.step()
            # Step 2: Train Feature Predictor and Label Predictor
            class_logits = self.model.label_predictor(feature1[:source_data.shape[0]])
            domain_logits = self.model.domain_classifier(feature1)
            loss = self.class_criterion(class_logits, source_label) - lamb * self.domain_criterion(domain_logits, domain_label1)
            running_F_loss+= loss.item()
            loss.backward()
            self.optimizer_F.step()
            self.optimizer_C.step()

            self.optimizer_D.zero_grad()
            self.optimizer_F.zero_grad()
            self.optimizer_C.zero_grad()

            total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
            total_num += source_data.shape[0]

        self.num_data = total_num

        return {
            'lr': lamb,
            'D loss': running_D_loss / (i+1), 
            'F loss': running_F_loss / (i+1), 
            'acc': total_hit / total_num,
        }

    @property
    def lr(self):
        return 1e-3

