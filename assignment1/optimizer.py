# Created by LIU Min
# 191240030@smail.nju.edu.cn

import torch
import torch.optim as optim

class LMOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model, args):
        self.lr = args.lr
        self._init_lr = args.lr
        self._min_lr = args.min_lr
        self._warmup_init_lr = args.warmup_init_lr
        self._warmup_epoches = args.warmup_epoches
        self._lr_scheduler = args.lr_scheduler

        self._last_epoch = -1
        # get optimizer
        if args.optimizer == 'Adam':
            self._optimizer = optim.Adam(
                model.parameters(), 
                betas = (0.9, 0.999),
                eps = 1e-8,
                weight_decay = args.weight_decay
            )
        elif args.optimizer == 'SGD':
            self._optimizer = optim.SGD(
                model.parameters(), 
                lr=args.warmup_init_lr, 
                momentum=0.9, 
                weight_decay=args.weight_decay
            )
        else:
            raise RuntimeError(f'Optimizer {args.optimizer} not supported yet.')
        # get optimizer
        self.lr = args.lr
        self.min_lr = args.min_lr

    def step(self, epoch):
        # cal lr for current epoch
        if epoch != self._last_epoch:
            self.lr = self.rate(epoch)
            self._last_epoch = epoch

        # set learning rate for optimizer
        for p in self._optimizer.param_groups:
            p['lr'] = self.lr

        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def rate(self, epoch):
        """ set rate during training
        """
        if epoch <= self._warmup_epoches:
            lrs = torch.linspace(self._warmup_init_lr, self._init_lr, self._warmup_epoches)
            return lrs[epoch - 1]
        else:
            if self._lr_scheduler == 'inverse_sqrt':
                return self._init_lr * (self._warmup_epoches ** 0.5) * (epoch ** -0.5)
            elif self._lr_scheduler == 'constant':
                return self._init_lr
            elif self._lr_scheduler == 'hand_design':
                pass







