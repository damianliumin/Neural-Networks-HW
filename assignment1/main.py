# Created by LIU Min
# 191240030@smail.nju.edu.cn

"""
    Overfitting is the major problem in this task. The following methods are used in this framework:
    - data augmentation: this is the most effective method
    - freeze some layers of pretrained model
    - dropout_ratio and weight_decay: tune the hyperparameters patiently
    
    Other methods to improve the performance:
    - ensemble learning (really useful)
    - more complex pretrained models (vgg seems better than resnet, but more complex ones are not necessary for this simple task)
"""

import os
import time
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import DataParallel
import torch.nn.functional as F

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

    if args.test_only == False:
        train(device, args)
    
    if args.train_only == False:
        test(device, args)


def train(device, args):
    # load data
    print('loading data...')
    tik = time.time()
    train_set = prep_dataloader(args.data, mode='train', batchsz=args.batch_size, num_workers=2, args=args)
    dev_set = prep_dataloader(args.data, mode='dev', batchsz=args.batch_size, args=args)
    tok = time.time()
    print(f'Train set: {len(train_set.dataset)} pictures.\n'
          f'Dev set: {len(dev_set.dataset)} pictures.\n'
           'Data loaded in {:.2f} s.'.format(tok - tik)
          )

    # load model
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
    trainer = Trainer(model, train_set, device, args)

    best_loss, best_epoch, best_acc = 1e9, -1, 0
    best_train_loss, best_train_epoch = 1e9, -1
    train_loss_rec, val_loss_rec = [], []
    while trainer.epoch <= args.max_epoches and trainer.lr >= args.min_lr :
        tik = time.time()
        train_state_dict = trainer.train_epoch()
        tok = time.time()
        pictps = trainer.num_data / (tok - tik)
        val_state_dict = validate(model, device, dev_set, trainer)

        print(
            'epoch {} | lr {:.4f} | spd {:.1f}p/s | tr_lss {:.3f} | tr_acc {:.2f}% | '
            'val_lss {:.3f} | val_acc {:.2f}%'.format(trainer.epoch, train_state_dict['lr'], pictps, train_state_dict['loss'], train_state_dict['acc'] * 100, val_state_dict['loss'], val_state_dict['acc'] * 100
            )
        )

        # save checkpoints (only when results are better on dev)
        if val_state_dict['acc'] > best_acc:
            save_checkpoint(model, trainer.epoch, train_state_dict, val_state_dict, args, True)
            best_loss = val_state_dict['loss']
            best_acc = val_state_dict['acc']
            best_epoch = trainer.epoch
        save_checkpoint(model, trainer.epoch, train_state_dict, val_state_dict, args, best=False)

        # update history
        if train_state_dict['loss'] < best_train_loss:
            best_train_loss = train_state_dict['loss']
            best_train_epoch = trainer.epoch

        train_loss_rec.append(train_state_dict['loss'])
        val_loss_rec.append(val_state_dict['loss'])

        # early stop
        if trainer.epoch - best_train_epoch >= args.early_stop and args.early_stop > 0:
            print('Early Stop!')
            break
    
    globaltok = time.time()

    # plot train / dev loss for debug
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        plt.plot(train_loss_rec, label='train')
        plt.plot(val_loss_rec, label='dev')
        plt.legend(loc='upper right')
        plt.savefig(args.log_dir + f'{args.model_name}.png', dpi=500)
    print('Training completed ({:.2f}s, {} epoches).'.format(globaltok - globaltik, trainer.epoch))
    print('Best epoch: {}, loss {:.3f}'.format(best_epoch, best_loss))


def validate(model, device, dataloader, trainer):
    num_correct = 0
    num_data = len(dataloader.dataset)
    total_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            bs, ncrops, c, h, w = np.shape(x)
            x = x.reshape(-1, c, h, w)
            x, y = x.to(device), y.to(device)
            out = model(x)
            out = out.reshape(bs, ncrops, -1).mean(1)
            loss = trainer.criterion(out, y)
            total_loss += loss.item() * len(x) / ncrops
            num_correct += np.sum((torch.argmax(F.softmax(out, dim=1), dim=1) == y).cpu().numpy())

    return {
        'loss': total_loss / num_data,
        'acc': num_correct / num_data
    }


def test(device, args):
    # load model
    if args.ensemble:
        # model_list = ['rsn1', 'rsn4', 'rsn5', 'rsn6']
        # model_list += ['vgg1', 'vgg2']
        model_list = eval(args.model_list)
        model = Ensemble(model_list, args)
    else:
        model = load_checkpoint(args.save_dir + f'{args.model_name}_checkpoint_last.pt')
    model = model.to(device)

    # load data
    test_set = prep_dataloader(args.data, mode='test', batchsz=args.batch_size, args=args)
    print(f'Testing on {len(test_set.dataset)} entries...')
    if args.test_output is None:
        raise RuntimeError('Please specify the output directory of testing.')
    if not os.path.exists(args.test_output):
        os.mkdir(args.test_output)
    f = open(args.test_output + model.name + '_output.csv', 'w')
    f.write('ID,emotion\n')
    id = 1
    with torch.no_grad():
        for x in test_set:
            bs, ncrops, c, h, w = np.shape(x)
            x = x.to(device)
            x = x.reshape(-1, c, h, w)
            out = model(x)
            out = out.reshape(bs, ncrops, -1).mean(1)
            y_pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            for y in y_pred:
                f.write(f'{id},{y}\n')
                id += 1
    f.close()
    print('Test completed.')


if __name__ == '__main__':
    main()


