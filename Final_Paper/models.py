from logging import setLogRecordFactory
import torch
import torch.nn as nn
import math
import random

class EnergyModel(nn.Module):
    def __init__(self, args):
        super(EnergyModel, self).__init__()
        self.embed_size = args.embed_size
        self.embed_mol = nn.Embedding(2, self.embed_size)
        self.embed_zdis = nn.Embedding(48, self.embed_size)
        
        self.conv1d = nn.Conv1d(self.embed_size, self.embed_size, 1)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.embed_size),
            nn.Linear(self.embed_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),
            nn.Linear(16, 8)
        )
        self.fc_a = nn.Sequential(
            nn.BatchNorm1d(self.embed_size),
            nn.Linear(self.embed_size, 1)
        )
        self.fc_b = nn.Sequential(
            nn.BatchNorm1d(self.embed_size),
            nn.Linear(self.embed_size, 1)
        )


    def forward(self, seq, zdis):
        """
        seq@Tensor(int): b x seqlen
        zdis@Tensor(float): b
        """
        x = self.embed_mol(seq) # b x seqlen x emb
        x = x.mean(1) # b x emb
        y = self.embed_zdis(zdis.to(torch.long)) # b x emb
        x += y

        a = self.fc_a(x).squeeze()
        b = self.fc_b(x).squeeze()
        x = self.fc(x)

        x = x.sum(1)
        x = (a * torch.LongTensor([seq.shape[1]]).to(x.device) + b) * x

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super(PositionalEncoding, self).__init__() 
        self.maxlen = max_len      
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        randst = random.randint(0, self.maxlen - len(x))
        return x + self.pe[randst:len(x)+randst, :]


class StructModel(nn.Module):
    def __init__(self, args):
        super(StructModel, self).__init__()
        self.embed_size = args.embed_size
        self.embed_mol = nn.Embedding(2, self.embed_size)
        self.embed_zdis = nn.Embedding(185, self.embed_size)

        self.positional_encoding = PositionalEncoding(self.embed_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.embed_size),
            nn.Linear(self.embed_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),
            nn.Linear(16, 8)
        )
        self.fc_a = nn.Sequential(
            nn.BatchNorm1d(self.embed_size),
            nn.Linear(self.embed_size, 1)
        )
        self.fc_b = nn.Sequential(
            nn.BatchNorm1d(self.embed_size),
            nn.Linear(self.embed_size, 1)
        )

    def forward(self, seq, zdis):
        """
        seq@Tensor(int): b x seqlen
        zdis@Tensor(float): b
        """
        # use transformer to cal state for each mol
        x = self.embed_mol(seq)         # b x seqlen x emb
        x = x.transpose(0, 1)           # seqlen x b x emb
        x = self.positional_encoding(x) # seqlen x b x emb
        x = self.encoder(x)             # seqlen x b x emb
        x = x.transpose(0, 1)           # b x seqlen x emb

        # consider global state
        global_state = x.mean(1).unsqueeze(1)                   # b x 1 x emb
        z = self.embed_zdis(zdis.to(torch.long)).unsqueeze(1)   # b x 1 x emb
        x += global_state + z                                   # b x seqlen x emb
        x = x.reshape(-1, self.embed_size)                      # * x emb

        a = self.fc_a(x).squeeze()
        b = self.fc_b(x).squeeze()
        x = self.fc(x)

        x = x.sum(1)
        x = (a * torch.LongTensor([seq.shape[1]]).to(x.device) + b) * x

        x = x.reshape(-1, seq.shape[1])

        return x


