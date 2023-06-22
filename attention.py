import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, d, n, l):
        super().__init__()
        self.d     = math.sqrt(d)
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
    def __init__(self, param, device):
        super().__init__()
        N, h, n, l    = param['dim']
        self.channels = param['channels']
        self.nb_class = param['nb_classes']
        self.t_in     = param['T_in']
        self.N        = N
        self.pos      = self.get_positional_encoding(device)
        self.multi    = nn.ModuleList()
        self.fc       = nn.ModuleList()
        self.norm1    = nn.ModuleList()
        self.norm2    = nn.ModuleList()
        for i in range(N):
            if i != 0:
                self.norm1.append(nn.LayerNorm(self.channels[i]))
            self.multi.append(MultiHead(h, self.channels[i], n, l))
            self.norm2.append(nn.LayerNorm(self.channels[i]))
            self.fc.append(nn.Linear(self.channels[i], self.channels[i+1]))
        self.cutoff = nn.parameter.Parameter(data=torch.zeros(param['T_out'],
                                                              param['nb_lon']*param['nb_lat'],
                                                              self.nb_class-1))
        nn.init.xavier_uniform_(self.cutoff)

    def get_positional_encoding(self, device):
        d = self.channels[0]
        if d % 2 != 0:
            raise RuntimeError
        pos = torch.zeros(self.t_in, d).to(device)
        for t in range(self.t_in):
            pos[t, :] = t + 1
        omega = torch.ones(1, d//2).to(device)
        for k in range(d//2):
            omega[0, k] = 1/1000**(2*k/d)
        omega = omega.repeat_interleave(2, dim=1)
        pos   = pos*omega
        phase = torch.tensor([0, torch.pi/2]).to(device)
        phase = phase.repeat(d//2).unsqueeze(0)
        pos   = torch.sin(pos + phase)
        return pos

    def forward(self, x, device):
        x = x + self.pos
        for i in range(self.N):
            if i != 0:
                x = self.norm1[i-1](x)
            x = self.multi[i](x)
            x = self.norm2[i](x)
            x = self.fc[i](x)
        probs = torch.zeros(*x.shape, self.nb_class).to(device)
        probs[..., 0]  = torch.sigmoid(self.cutoff[..., 0] - x)
        probs[..., -1] = 1 - torch.sigmoid(self.cutoff[..., -1] - x)
        for i in range(1, self.nb_class - 1):
            probs[..., i] = torch.sigmoid(self.cutoff[..., i] - x) \
                            - torch.sigmoid(self.cutoff[..., i-1] - x)
        return probs


def main():
    d    = 6
    t_in = 8
    if d % 2 != 0:
        raise RuntimeError
    pos = torch.zeros(t_in, d)
    for t in range(t_in):
        pos[t, :] = t + 1
    print(pos)
    omega = torch.ones(1, d // 2)
    for k in range(d // 2):
        omega[0, k] = 1 / 1000 ** (2 * k / d)
    omega = omega.repeat_interleave(2, dim=1)
    print(f'omega: shape={omega.shape}\n{omega}')
    pos = pos * omega
    print(pos)
    phase = torch.tensor([0, torch.pi / 2])
    phase = phase.repeat(d // 2).unsqueeze(0)
    print(f'phase: shape={phase.shape}\n{phase}')
    pos = pos + phase
    print(pos)
    pos = torch.sin(pos)
    print(f'pos: shape={pos.shape}\n{pos}')


if __name__ == '__main__':
    main()