# Created by LIU Min
# 191240030@smail.nju.edu.cn

import argparse

def get_parse():
    parser = argparse.ArgumentParser()
    # path, dir, files
    parser.add_argument('data', metavar='PATH', help='path of the dataset')
    parser.add_argument('--model-name', required=True, metavar='NAME', help='name of the modle')
    parser.add_argument('--model-type', required=True, metavar='CLS', help='class of the model')
    parser.add_argument('--save-dir', required=True, metavar='DIR', help='checkpoints')
    parser.add_argument('--log-dir', metavar='DIR', help='log')
    parser.add_argument('--test-output', metavar='DIR', help='test output')

    # data
    parser.add_argument('--num-classes', metavar='metavar', default=0, type=int, help='number of classes')
    parser.add_argument('--cache', action='store_true', help='accelerate data loading process')

    # model / training settings
    parser.add_argument('--pretrained', action='store_true', help='use pretrained models')
    
    parser.add_argument('--max-epoches', metavar='ME', default=150, type=int, help='max number of epoches')
    parser.add_argument('--max-updates', metavar='ME', default=150, type=int, help='max number of updates')
    parser.add_argument('--batch-size', metavar='BSZ', default=64, type=int, help='batch size')
    parser.add_argument('--dropout-ratio', metavar='D', default=0.0, type=float, help='dropout ratio of Dropout Layers')
    parser.add_argument('--seed', metavar='S', default=0, type=int, help='random seed')

    parser.add_argument('--lr', metavar='LR', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--warmup-init-lr', metavar='I', default=1e-7, type=float, help='initial learning rate for warm up')
    parser.add_argument('--warmup-epoches', metavar='U', default=0, type=int, help='number of epoches for warm up')
    parser.add_argument('--min-lr', metavar='M', default=1e-9, type=float, help='minimal learning rate')
    parser.add_argument('--lr-scheduler', metavar='LRS', default='inverse_sqrt', help='learning rate scheduler scheme')

    parser.add_argument('--optimizer', metavar='OPM', default='Adam', help='specify optimizer')
    parser.add_argument('--weight-decay', default=0.0, type=float, metavar='WD', help='weight decay')

    parser.add_argument('--early-stop', default=-1, type=int, metavar='S', help='stop when loss no longer decreases')

    parser.add_argument('--train-only', action='store_true', help='train only')
    parser.add_argument('--test-only', action='store_true', help='test only')

    # ensemble
    parser.add_argument('--ensemble', action='store_true', help='test using ensembled models')
    parser.add_argument('--model-list', default=None, metavar='ML', help='list of models')

    # device
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    parser.add_argument('--world-size', metavar='N', default=1, type=int, help='number of GPU to use')
    parser.add_argument('--device-id', metavar='ID', default=0, type=int, help='which GPU to use')
    

    return parser


if __name__ == '__main__':
    # test and debug
    parser = get_parse()
    parser.print_help()
    

