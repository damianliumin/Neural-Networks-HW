from struct import Struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EnergyData(Dataset):
    def __init__(self, split='train', seqlen=6):
        self.split = split
        if split == 'test':
            if seqlen == 20:
                self.seq = torch.LongTensor(np.loadtxt('data/20pred.csv', delimiter=','))
            else:
                self.seq = torch.LongTensor(np.loadtxt(f'data/{seqlen}/sq.txt'))
        else:
            self.seq = torch.LongTensor(np.loadtxt(f'data/{seqlen}/sq.txt'))
            self.energy = torch.Tensor(np.zeros((0, 2))).to(torch.float32)
            for i in range(1, len(self.seq)+1):
                tmp = torch.FloatTensor(np.loadtxt(f'data/{seqlen}/FU_{i}.txt'))
                self.energy = torch.vstack((self.energy, tmp))
            print(self.energy.mean(), self.energy.std())

    def __len__(self):
        return len(self.seq) * 48

    def __getitem__(self, idx):
        """ Return sequence, Z distance, and (F, U) """
        if self.split != 'test':
            return self.seq[idx // 48], idx % 48, self.energy[idx]
        else:
            return self.seq[idx // 48], idx % 48

def prep_energy_dataloader(seqlen=6, split='train', batchsz=32):
    energydataset = EnergyData(split, seqlen)
    return DataLoader(
        energydataset, batchsz,
        shuffle=(split=='train'),
        num_workers=2
    )

class StructData(Dataset):
    def __init__(self, split='train', seqlen=6):
        self.split = split
        self.seqlen = seqlen
        if split == 'test':
            if seqlen == 20:
                self.seq = torch.LongTensor(np.loadtxt('data/20pred.csv', delimiter=','))
            else:
                self.seq = torch.LongTensor(np.loadtxt(f'data/{seqlen}/sq.txt'))  
        else:
            self.seq = torch.LongTensor(np.loadtxt(f'data/{seqlen}/sq.txt'))
            self.prob = torch.Tensor(np.zeros((0, self.seqlen))).to(torch.float32)
            for i in range(1, len(self.seq)+1):
                tmp = torch.FloatTensor(np.loadtxt(f'data/{seqlen}/Structure_{i}_interp.txt'))[:185]
                self.prob = torch.vstack((self.prob, tmp))
            self.prob = self.prob * 10000 - 1

    def __len__(self):
        return len(self.seq) * 185

    def __getitem__(self, idx):
        """ return sequence, Z distance, prob """
        if self.split != 'test':
            return self.seq[idx // 185], idx % 185, self.prob[idx]
        else:
            return self.seq[idx // 185], idx % 185

def prep_struct_dataloader(seqlen=6, split='train', batchsz=32):
    structdataset = StructData(split, seqlen)
    return DataLoader(
        structdataset, batchsz,
        shuffle=(split!='test'),
        num_workers=2
    )

if __name__=='__main__':
    # dataloader = prep_struct_dataloader(6, batchsz=32, split='train')
    # for i in dataloader:
    #     print(i)
    #     assert False
    dataloader = prep_energy_dataloader(6, batchsz=32, split='train')
    dataloader = prep_energy_dataloader(8, batchsz=32, split='train')
    dataloader = prep_energy_dataloader(10, batchsz=32, split='train')
    dataloader = prep_energy_dataloader(12, batchsz=32, split='train')