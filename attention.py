import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, d, n, l):
        super().__init__()
        self.d     = torch.sqrt(d)
        self.query = nn.Linear(d, n, bias=False)
        self.key   = nn.Linear(d, n, bias=False)
        self.value = nn.Linear(d, l, bias=False)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        a = F.softmax(torch.bmm(q, k.transpose(1, 2))/self.d, dim=2)
        return torch.bmm(a, v)


class MultiHead(nn.Module):
    """
    Multi Head Attention with skip connection
    """
    def __init__(self, h, d, n, l):
        super().__init__()
        self.heads = nn.ModuleList()
        for i in range(h):
            self.heads.append(AttentionHead(d, n, l))
        self.weights = nn.Linear(l*h, d, bias=False)

    def forward(self, x):
        res = []
        for head in self.heads:
            res.append(head(x))
        return self.weights(torch.cat(res, dim=2)) + x


class CongTrans(nn.Module):
    def __init__(self, param):
        super().__init__()
        N, h, d, n, l = param['dim']
        channels      = param['channels']
        self.nb_class = param['nb_classes']
        self.N        = N
        self.multi    = nn.ModuleList()
        self.fc       = nn.ModuleList()
        for i in range(N):
            self.multi.append(MultiHead(h, channels[i], n, l))
            self.fc.append(nn.Linear(channels[i], channels[i+1]))
        self.cutoff = nn.parameter.Parameter(data=torch.zeros(param['T_out'],
                                                              param['nb_lon']*param['nb_lat'],
                                                              self.nb_class-1))
        nn.init.xavier_uniform_(self.cutoff)

    def forward(self, x):
        for i in range(self.N):
            x = self.multi(x)
            x = self.fc(x)
        probs = torch.zeros(*x.shape, self.nb_class)
        probs[..., 0]  = torch.sigmoid(self.cutoff[..., 0] - x)
        probs[..., -1] = 1 - torch.sigmoid(self.cutoff[..., -1] - x)
        for i in range(1, self.nb_class - 1):
            probs[..., i] = torch.sigmoid(self.cutoff[..., i] - x) \
                            - torch.sigmoid(self.cutoff[..., i-1] - x)
        return probs
