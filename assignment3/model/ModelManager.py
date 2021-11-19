# Created by LIU Min
# 191240030@smail.nju.edu.cn

import os
from model.DaNN import DaNN

import torch

def get_model(args):
    """ Get model according to args (only used when training)
    """
    if args.model_type == 'DaNN':
        model = DaNN(args.model_name, args.model_type, args.dropout_ratio)
    else:
        raise RuntimeError(f'Unknown model type: {args.model_type}')
    return model

def save_checkpoint(model, epoch, train_state, args, best=False, name=None):
    """ Save checkpoint for best / last model
    """
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if name is None:
        path = args.save_dir + f'{model.name}_checkpoint_last.pt'
    else:
        path = args.save_dir + f'{name}_checkpoint_last.pt'

    torch.save({
        'epoch': epoch,
        'train_state': train_state,
        'state_dict': model.state_dict(),
        'model_type': model.model_type,
        'model_name': model.name,
    }, path)
        
def load_checkpoint(path):
    """ Load checkpoint for best / last model
    """
    if not os.path.exists(path):
        return None, None
    states = torch.load(path, map_location='cuda:0')
    epoch = states['epoch']
    train_state = states['train_state']
    model_type = states['model_type']
    model_name = states['model_name']

    if model_type == 'DaNN':
        model = DaNN(model_name, model_type)
    else:
        assert False

    model.load_state_dict(states['state_dict'])
    print('Load checkpoint on epoch {} with: D loss {:.3f} | F loss {:.3f} | acc {:.2f}%'.format(epoch, train_state['D loss'], train_state['F loss'], train_state['acc'] * 100))

    return model, epoch


