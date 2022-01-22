from concurrent.futures.process import _process_worker
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
from data import prep_energy_dataloader, prep_struct_dataloader
from models import EnergyModel, StructModel


def main(args):
    # load data
    if not args.test_only:
        batchsz = 128
        validset = None
        if args.task in ('F', 'U'):
            datasets = [prep_energy_dataloader(i, 'train', batchsz) for i in (6, 8, 10, 12)]
            if not args.no_dev:
                validset = prep_energy_dataloader(seqlen=12, split='valid', batchsz=batchsz)
        else:
            datasets = [prep_struct_dataloader(i, 'train', batchsz) for i in (6, 8, 12)]
            if not args.no_dev:
                validset = prep_struct_dataloader(seqlen=8, split='valid', batchsz=batchsz)
    if not args.train_only:
        if args.task in ('F', 'U'):
            testset = prep_energy_dataloader(seqlen=20, split='test', batchsz=48)
        else:
            testset = prep_struct_dataloader(seqlen=20, split='test', batchsz=185)
    print('Successfully loaded data!')

    # load model
    model = EnergyModel(args).to(args.device) if args.task != 'Stru' else StructModel(args).to(args.device)

    # train model
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.01**(1/args.epochs))

    if not args.test_only:
        if args.task in ('F', 'U'):
            train_energy(args, model, optimizer, criterion, scheduler, datasets, validset)
        else:
            train_struct(args, model, optimizer, criterion, scheduler, datasets, validset)
        torch.save(model, f'models/{args.name}')
    if not args.train_only:
        model = torch.load(f'models/{args.name}').to(args.device)
        if args.task in ('F', 'U'):
            test_energy(args, model, testset)
        else:
            test_struct(args, model, testset)

def train_energy(args, model, optimizer, criterion, scheduler, datasets, validset):
    def validate():
        loss_dev = 0
        with torch.no_grad():
            for seq, zdis, energy in validset:
                seq, zdis, energy = seq.to(device), zdis.to(device).to(torch.float32), energy.to(device)
                energy = energy[:, 0] if args.task == 'F' else energy[:, 1]
                energy_pred = model(seq, zdis)
                loss = criterion(energy_pred, energy)
                loss_dev += loss.item()
            loss_dev = loss_dev / len(validset)
            for i in range(5):
                print("{:7.4f}, {:7.4f}".format(energy[-i-1], energy_pred[-i-1]))
        return loss_dev
                
    device = args.device
    model.train()
    loss_train_rec = [[] for _ in range(len(datasets))]
    loss_dev_rec = []
    for epoch in range(1, args.epochs+1):
        loss_train_dataset = []
        print('Training...')
        for id, dataset in enumerate(datasets):
            loss_train = 0
            for seq, zdis, energy in dataset:
                seq, zdis, energy = seq.to(device), zdis.to(device).to(torch.float32), energy.to(device)
                energy = energy[:, 0] if args.task == 'F' else energy[:, 1]
                energy_pred = model(seq, zdis)
                loss = criterion(energy_pred, energy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            loss_train = loss_train / len(dataset)
            loss_train_dataset.append(loss_train)
            print(f'dataset {id}:')
            for i in range(5):
                print("{:7.4f}, {:7.4f}".format(energy[-i-1], energy_pred[-i-1]))
            
        scheduler.step()
        # validate
        if not args.no_dev:
            model.eval()
            print('Validating...')
            loss_dev = validate()
            model.train()
        else:
            loss_dev = 0
        # epoch summary
        print('epoch {} | loss '.format(epoch), end='')
        for loss in loss_train_dataset:
            print('{:.4f}'.format(loss), end=' ')
        print('| dev_loss {:.4f}'.format(loss_dev))
        # record loss for log
        for (rec, loss) in zip(loss_train_rec, loss_train_dataset):
            rec.append(loss)
        loss_dev_rec.append(loss_dev)

        # print(model.a, model.b)
        print('++++++++++++++++++++++++++')

    plt.figure()
    plt.ylim(0, 3.5)
    for i in range(len(datasets)):
        print(i, len(loss_train_rec))
        plt.plot(loss_train_rec[i], label='train {}'.format(i))
    plt.plot(loss_dev_rec, label='dev')
    plt.legend(loc='upper right')
    plt.savefig(f'log/{args.name}_loss.png', dpi=500)
    plt.close()

def test_energy(args, model, testset):
    def get_seq_name(seq):
        s = ''
        for c in seq[0]:
            s += str(c.item())
        return s

    def process_results(energy):
        energy = torch.hstack([energy, torch.zeros(1).to(energy.device)])
        energy = (energy + energy.flip(0)) / 2
        return energy[:-1]

    model.eval()
    device = args.device
    results = np.zeros((0, 48))
    with torch.no_grad():
        for seq, zdis in testset:
            seq, zdis = seq.to(device), zdis.to(device).to(torch.float32)
            energy_pred = model(seq, zdis)
            energy_pred = process_results(energy_pred)
            results = np.vstack((results, energy_pred.unsqueeze(0).cpu().numpy()))
            energy_pred = list(energy_pred)
            seqname = get_seq_name(seq)
            plt.figure()
            plt.title(seqname)
            plt.plot(energy_pred)
            plt.savefig(f'figs/{args.task}_{seqname}.png', dpi=300)
            plt.close()
            print(f'Tested on {seqname}.')
    np.savetxt(f'results/{args.task}_predict.csv', results.transpose(), delimiter=',')

def train_struct(args, model, optimizer, criterion, scheduler, datasets, validset):
    def validate():
        loss_dev = 0
        with torch.no_grad():
            for seq, zdis, prob in validset:
                seq, zdis, prob = seq.to(device), zdis.to(device).to(torch.float32), prob.to(device)
                prob_pred = model(seq, zdis)
                loss = criterion(prob_pred, prob)
                loss_dev += loss.item()
            loss_dev = loss_dev / len(validset)
            print(prob[0], prob_pred[0], sep='\n')
        return loss_dev
                
    device = args.device
    model.train()
    loss_train_rec = [[] for _ in range(len(datasets))]
    loss_dev_rec = []
    for epoch in range(1, args.epochs+1):
        loss_train_dataset = []
        print('Training...')
        for id, dataset in enumerate(datasets):
            loss_train = 0
            for seq, zdis, prob in dataset:
                seq, zdis, prob = seq.to(device), zdis.to(device).to(torch.float32), prob.to(device)
                prob_pred = model(seq, zdis)
                loss = criterion(prob_pred, prob)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            loss_train = loss_train / len(dataset)
            loss_train_dataset.append(loss_train)
            print(f'dataset {id}:')
            print(prob[0], prob_pred[0], sep='\n')
        scheduler.step()
        # validate
        if not args.no_dev:
            model.eval()
            print('Validating...')
            loss_dev = validate()
            model.train()
        else:
            loss_dev = 0
        # epoch summary
        print('epoch {} | loss '.format(epoch), end='')
        for loss in loss_train_dataset:
            print('{:.4f}'.format(loss), end=' ')
        print('| dev_loss {:.4f}'.format(loss_dev))
        # record loss for log
        for (rec, loss) in zip(loss_train_rec, loss_train_dataset):
            rec.append(loss)
        loss_dev_rec.append(loss_dev)

        # print(model.a, model.b)
        print('++++++++++++++++++++++++++')

    plt.figure()
    plt.ylim(0, 3.5)
    for i in range(len(datasets)):
        print(i, len(loss_train_rec))
        plt.plot(loss_train_rec[i], label='train {}'.format(i))
    plt.plot(loss_dev_rec, label='dev')
    plt.legend(loc='upper right')
    plt.savefig(f'log/{args.name}_loss.png', dpi=500)
    plt.close()

def test_struct(args, model, testset):
    def get_seq_name(seq):
        s = ''
        for c in seq[0]:
            s += str(c.item())
        return s

    def process_results(prob):
        """ prob: 185 x seqlen """
        x, y = prob[0, :].unsqueeze(0), prob[-1, :].unsqueeze(0)
        prob = torch.vstack((x, prob, y))
        prob = prob.T.unsqueeze(0) # 1 x seqlen x 189
        prob = F.avg_pool1d(prob, kernel_size=3, stride=1, padding=0)
        prob = (prob + 1) / 10000
        prob = prob.squeeze().T # 185 x seqlen
        prob = prob.clamp(0, 1)
        return prob

    model.eval()
    device = args.device
    results = np.zeros((480, 0))
    with torch.no_grad():
        for seq, zdis in testset:
            seq, zdis = seq.to(device), zdis.to(device).to(torch.float32)
            prob_pred = model(seq, zdis) # 185 x seqlen
            prob_pred = process_results(prob_pred)
            tmp = torch.hstack((prob_pred.mean(1), torch.zeros(480-185).to(prob_pred.device))).unsqueeze(1) # 480 x 1
            results = np.hstack((results, tmp.cpu().numpy()))
            seqname = get_seq_name(seq)
            plt.figure(figsize=(5, 6))
            plt.title(seqname)
            plt.ylim(0, 0.0006)
            plt.xlabel('R distance')
            for i in range(prob_pred.shape[-1]):
                plt.plot(np.arange(185) / 10, prob_pred[:, i].cpu().numpy(), label=f'{i+1}', linewidth=1)
            plt.legend(loc='upper right')
            plt.savefig(f'figs/{args.task}_{seqname}.png', dpi=400)
            plt.close()
            print(f'Tested on {seqname}.')
        np.savetxt(f'results/{args.task}_predict.csv', results, delimiter=',')


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', metavar='N', help='name of the model')
    parser.add_argument('--task', metavar='T', default='F', choices=('F', 'U', 'Stru'), help='specify task')
    parser.add_argument('--embed-size', metavar='ES', type=int, default=16, help='embedding size in models')
    parser.add_argument('--lr', metavar='LR', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--epochs', metavar='E', type=int, default=100, help='number of epochs')
    parser.add_argument('--no-dev', action='store_true', help='not use dev set when training')
    parser.add_argument('--train-only', action='store_true', help='train only')
    parser.add_argument('--test-only', action='store_true', help='test only')
    
    return parser


if __name__=='__main__':
    args = get_parse().parse_args()
    args.device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    print(args)

    main(args)
