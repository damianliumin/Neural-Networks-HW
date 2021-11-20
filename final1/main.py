# Created by LIU Min
# 191240030@smail.nju.edu.cn


import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.nn import DataParallel
from torch.autograd import Variable
import torchvision

from data import prep_dataloader
from model.Ensemble import Ensemble
from options import get_parse
from trainer import Trainer

from model.ModelManager import load_checkpoint, save_checkpoint, get_model


def main():
    # parse and print args
    args = get_parse().parse_args()
    print(args)
    
    # set random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # set device
    if torch.cuda.is_available() and not args.cpu:
        device = f'cuda:{args.device_id}'
    else:
        device = 'cpu'

    if args.train_only:
        train(device, args)
    elif args.test_only:
        test(device, args)
    else:
        assert False


def train(device, args):
    # load data
    print('loading data...')
    tik = time.time()
    dataloader = prep_dataloader(args.data, mode='train', batchsz=args.batch_size, num_workers=2)
    tok = time.time()
    print(f'Dataset: {len(dataloader.dataset)} pictures.\n'
           'Data loaded in {:.2f} s.'.format(tok - tik)
          )

    # load model
    model, start_epoch = None, None
    if not args.retrain:
        model, start_epoch = load_checkpoint(args.save_dir + f'{args.model_name}_checkpoint_last.pt')
    if model is None:
        model = get_model(args)
    print(model)

    if not args.cpu:
        if args.world_size > 1:
            # multi-GPU
            device_cnt = torch.cuda.device_count()
            if args.world_size > device_cnt:
                raise RuntimeError('World size should no be larger than number of GPUs.')
            device_ids = [i % device_cnt for i in range(args.device_id, args.device_id + args.world_size)]
            model = DataParallel(model, device_ids)
            model = model.to(device)
            raise RuntimeWarning('WARNING | Multi-GPU has not been implemented correctly!')
        else:
            model = model.to(device)
        print(f'Number of GPU: {args.world_size}. Ouput ID: {args.device_id}')
    else:
        print('Training on CPU.')
    
    # training epoch
    globaltik = time.time()
    trainer = Trainer(model, dataloader, device, args)
    trainer.epoch = start_epoch if start_epoch is not None else 0
    z_sample = Variable(torch.randn(100, 100)).to(device)

    d_loss, g_loss = [], []
    while trainer.epoch < args.max_epoches and trainer.lr >= args.min_lr :
        tik = time.time()
        train_state_dict = trainer.train_epoch()
        tok = time.time()
        pictps = trainer.num_data / (tok - tik)

        print(
            'epoch {} | steps {} | spd {:.1f}p/s | D loss {:.3f} | G loss {:.2f}'.format(trainer.epoch, trainer.steps, pictps, train_state_dict['D loss'], train_state_dict['G loss'])
        )

        d_loss.append(train_state_dict['D loss'])
        g_loss.append(train_state_dict['G loss'])

        validate(device, model, z_sample, trainer.epoch, args)

        # save checkpoints (only when results are better on dev)
        save_checkpoint(model, trainer.epoch, train_state_dict, args, best=False)
    
    globaltok = time.time()

    # plot train / dev loss for debug
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        plt.plot(d_loss, label='D loss')
        plt.plot(g_loss, label='G loss')
        plt.legend(loc='upper right')
        plt.savefig(args.log_dir + f'{args.model_name}.png', dpi=500)
    print('Training completed ({:.2f}s, {} epoches).'.format(globaltok - globaltik, trainer.epoch))


def validate(device, model, z_sample, epoch, args):
    model.generator.eval()

    f_imgs_sample = (model.generator(z_sample.to(device)).data + 1) / 2.0
    filename = os.path.join(args.log_dir, f'{model.name}_epoch_{epoch:03d}.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    print(f' | Save some samples to {filename}.')


def test(device, args):
    model, _ = load_checkpoint(args.save_dir + f'{args.model_name}_checkpoint_last.pt')
    model = model.to(device)
    model.generator.eval()

    z_sample = Variable(torch.randn(100, 100))
    f_imgs_sample = (model.generator(z_sample.to(device)).data + 1) / 2.0
    filename = os.path.join(args.test_output, f'{model.name}.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    print(f' | Save some samples to {filename}.')

    print('Test completed.')


    

if __name__ == '__main__':
    main()

