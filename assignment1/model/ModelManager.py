# Created by LIU Min
# 191240030@smail.nju.edu.cn

import os
from model.LMResNet import LMResNet
from model.ResNet import ResNet
from model.DeepEmotion import DeepEmotion
from model.VGG import VGG
from model.Inception import Inception
from model.ResNext import ResNext

import torch

def get_model(args):
    """ Get model according to args (only used when training)
    """
    if args.model_type.startswith('resnet'):
        model = ResNet(args.model_name, args.model_type, args.dropout_ratio, args.pretrained)
    elif args.model_type.startswith('vgg'):
        model = VGG(args.model_name, args.model_type, args.dropout_ratio, args.pretrained)
    elif args.model_type == 'lmresnet':
        model = LMResNet(args.model_name, args.model_type, args.dropout_ratio)
    elif args.model_type == 'deepemotion':
        model = DeepEmotion(args.model_name, args.model_type, args.dropout_ratio)
    elif args.model_type.startswith('resnext'):
        model = ResNext(args.model_name, args.model_type, args.dropout_ratio)
    elif args.model_type.startswith('inception'):
        model = Inception(args.model_name, args.model_type, args.dropout_ratio)
    else:
        raise RuntimeError(f'Unknown model type: {args.model_type}')
    return model

def save_checkpoint(model, epoch, train_state, dev_state, args, best=False):
    """ Save checkpoint for best / last model
    """
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if best:
        path = args.save_dir + f'{model.name}_checkpoint_best.pt'
        print('Save best checkpoint on epoch {} with loss {:.3f}'.format(epoch, dev_state['loss']))
    else:
        path = args.save_dir + f'{model.name}_checkpoint_last.pt'

    torch.save({
        'epoch': epoch,
        'train_state': train_state,
        'dev_state': dev_state,
        'state_dict': model.state_dict(),
        'model_type': model.model_type,
        'model_name': model.name,
    }, path)
        
def load_checkpoint(path):
    """ Load checkpoint for best / last model
    """
    states = torch.load(path, map_location='cuda:2')
    epoch = states['epoch']
    train_state = states['train_state']
    dev_state = states['dev_state']
    model_type = states['model_type']
    model_name = states['model_name']

    if model_type.startswith('resnet'):
        model = ResNet(model_name, model_type)
    elif model_type.startswith('vgg'):
        model = VGG(model_name, model_type)
    elif model_type == 'lmresnet':
        model = LMResNet(model_name, model_type)
    elif model_type == 'deepemotion':
        model = DeepEmotion(model_name, model_type)
    elif model_type.startswith('resnext'):
        model = ResNext(model_name, model_type)
    elif model_type.startswith('inception'):
        model = Inception(model_name, model_type)
    else:
        raise RuntimeError(f'Unknown model type: {model_type}')

    model.load_state_dict(states['state_dict'])
    print('Load checkpoint on epoch {} with\ndev loss {:.3f} | dev acc {:.2f}% | train loss {:.3f} | train acc {:.2f}%'.format(epoch, dev_state['loss'], dev_state['acc'] * 100, train_state['loss'], train_state['acc'] * 100))

    return model


