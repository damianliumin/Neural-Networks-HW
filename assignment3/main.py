# Created by LIU Min
# 191240030@smail.nju.edu.cn


import os
import time
from collections import Counter
import numpy as np
from numpy.lib.utils import source
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import DataParallel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data import prep_dataloader
from model.Ensemble import Ensemble
from options import get_parse
from trainer import Trainer
from optimizer import LMOpt

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
    elif args.analysis:
        analysis(device, args)
    else:
        assert False


def train(device, args):
    # load data
    print('loading data...')
    tik = time.time()
    source_dataset = prep_dataloader(args.src, mode='train', batchsz=args.batch_size, num_workers=2)
    target_dataset = prep_dataloader(args.tgt, mode='train', batchsz=args.batch_size, num_workers=2)
    tok = time.time()
    print(f'Source dataset: {len(source_dataset.dataset)} pictures.\n'
          f'Target dataset: {len(target_dataset.dataset)} pictures.\n'
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
    trainer = Trainer(model, source_dataset, target_dataset, device, args)
    trainer.epoch = start_epoch if start_epoch is not None else 0

    d_loss, f_loss = [], []
    while trainer.epoch < args.max_epoches and trainer.lr >= args.min_lr :
        tik = time.time()
        train_state_dict = trainer.train_epoch()
        tok = time.time()
        pictps = trainer.num_data / (tok - tik)

        print(
            'epoch {} | lr {:.6f} | spd {:.1f}p/s | D loss {:.3f} | F loss {:.2f} | '
            'acc {:.3f}%'.format(trainer.epoch, train_state_dict['lr'], pictps, train_state_dict['D loss'], train_state_dict['F loss'], train_state_dict['acc'] * 100
            )
        )

        d_loss.append(train_state_dict['D loss'])
        f_loss.append(train_state_dict['F loss'])

        # save checkpoints (only when results are better on dev)
        save_checkpoint(model, trainer.epoch, train_state_dict, args, best=False)
    
    globaltok = time.time()

    # plot train / dev loss for debug
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        plt.plot(d_loss, label='D loss')
        plt.plot(f_loss, label='F loss')
        plt.legend(loc='upper right')
        plt.savefig(args.log_dir + f'{args.model_name}.png', dpi=500)
    print('Training completed ({:.2f}s, {} epoches).'.format(globaltok - globaltik, trainer.epoch))


def test(device, args):
    # load model
    if args.ensemble:
        # model_list = ['rsn1', 'rsn4', 'rsn5', 'rsn6']
        model_list = eval(args.model_list)
        model = Ensemble(model_list, args)
    else:
        model, _ = load_checkpoint(args.save_dir + f'{args.model_name}_checkpoint_last.pt')
    model = model.to(device)
    model.eval()

    # load data
    test_set = prep_dataloader(args.tgt, mode='test', batchsz=args.batch_size, num_workers=1)

    print(f'Testing on {len(test_set.dataset)} entries...')
    if args.test_output is None:
        raise RuntimeError('Please specify the output directory of testing.')
    if not os.path.exists(args.test_output):
        os.mkdir(args.test_output)
    
    results = []
    with torch.no_grad():
        for x, _ in test_set:
            x = x.to(device)
            out = model(x)
            y_pred = torch.argmax(out, dim=1).cpu().detach().numpy()
            results.append(y_pred)
    
    results = np.concatenate(results)
    df = pd.DataFrame({'id': np.arange(0, len(results)), 'label': results})
    df.to_csv(f'{args.test_output + args.model_name}.csv',index=False)
    print('Test completed.')

def analysis(device, args):
    model, _ = load_checkpoint('checkpoints/dann3_checkpoint_last.pt')
    model = model.to(device)
    model.eval()
    feature_extractor = model.feature_extractor
    print('Model loaded!')
    
    source_dataloader = prep_dataloader(args.src, mode='train', batchsz=64, num_workers=2)
    target_dataloader = prep_dataloader(args.tgt, mode='train', batchsz=64, num_workers=2)
    print('Data loaded!')

    src, tgt = [], []
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data, target_data = source_data.to(device), target_data.to(device)
        feature_src = feature_extractor(source_data)
        src.append(feature_src.detach().cpu())
        feature_tgt = feature_extractor(target_data)
        tgt.append(feature_tgt.detach().cpu())
    print('Features generated!')

    src = torch.cat(src).cpu()
    tgt = torch.cat(tgt).cpu()
    assert src.shape[1] == 32
    X = torch.cat([src, tgt])
    tsne = TSNE(n_components=2)
    newX = tsne.fit_transform(X.numpy())
    print('PCA finished!')
    assert newX.shape[1] == 2
    newsrc = newX[:src.shape[0]].transpose()
    newtgt = newX[src.shape[0]:].transpose()
    
    plt.axis('off')
    plt.scatter(newsrc[0], newsrc[1], s=0.1, c='r', label='source')
    plt.scatter(newtgt[0], newtgt[1], s=0.1, c='b', label='target')
    # plt.legend(loc='upper right')
    plt.savefig('adversal.png', dpi=500)
    

if __name__ == '__main__':
    main()

