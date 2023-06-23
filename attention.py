import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_positional_encoding(d, t_in, device):
    if d % 2 != 0:
        raise RuntimeError
    pos = torch.zeros(t_in, d).to(device)
    for t in range(t_in):
        pos[t, :] = t + 1
    omega = torch.ones(1, d // 2).to(device)
    for k in range(d // 2):
        omega[0, k] = 1 / 1000 ** (2 * k / d)
    omega = omega.repeat_interleave(2, dim=1)
    pos = pos * omega
    phase = torch.tensor([0, torch.pi / 2]).to(device)
    phase = phase.repeat(d // 2).unsqueeze(0)
    pos = torch.sin(pos + phase)
    return pos


class AttentionHead(nn.Module):
    """
    Multi Head Attention with skip connection
    h: nb of heads
    d: input dimension
    """
    def __init__(self, h, d, device):
        super().__init__()
        if d % h != 0:
            raise RuntimeError
        self.rd     = math.sqrt(d)
        self.h      = h
        self.device = device
        self.query  = nn.Linear(d, h*(d//h), bias=False)
        self.key    = nn.Linear(d, h*(d//h), bias=False)
        self.value  = nn.Linear(d, h*(d//h), bias=False)

    def forward(self, x, masking=False):
        q = self.query(x)  # b x l x h.s where s = d//h
        q = q.view(q.shape[0], q.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        k = self.key(x)  # b x l x h.s
        k = k.view(k.shape[0], k.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        v = self.value(x)  # b x l x h.s
        v = v.view(v.shape[0], v.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        a = torch.matmul(q, k.transpose(2, 3))/self.rd  # b x h x l x l
        if masking:
            mask = torch.ones(*a.shape).triu(diagonal=1).to(self.device)
            mask = mask.masked_fill(mask==1, float('-inf'))
            a    = a + mask
        a = F.softmax(a, dim=3)  # b x h x l x l
        a = torch.matmul(a, v).transpose(1, 2)  # b x l x h x s
        return x + a.reshape(a.shape[0], a.shape[1], -1)  # b x l x h.s


class FinalHead(nn.Module):
    """
    Last Multi Head of the encoder
    """
    def __init__(self, h, d):
        super().__init__()
        if d % h != 0:
            raise RuntimeError
        self.h     = h
        self.key   = nn.Linear(d, h*(d//h), bias=False)
        self.value = nn.Linear(d, h*(d//h), bias=False)

    def forward(self, x):
        k = self.key(x)
        k = k.view(k.shape[0], k.shape[1], self.h, -1).transpose(1, 2)
        v = self.value(x)
        v = v.view(v.shape[0], v.shape[1], self.h, -1).transpose(1, 2)
        return k, v


class DecoderHead(nn.Module):
    """
    Multi Head Attention using key and value from the encoder
    """
    def __init__(self, h, d, device):
        super().__init__()
        if d % h != 0:
            raise RuntimeError
        self.rd     = math.sqrt(d)
        self.h      = h
        self.device = device
        self.query  = nn.Linear(d, h*(d//h), bias=False)

    def forward(self, x, k, v):
        q = self.query(x)
        q = q.view(q.shape[0], q.shape[1], self.h, -1).transpose(1, 2)
        a = torch.matmul(q, k.transpose(2, 3))/self.rd
        mask = torch.ones(*a.shape).triu(diagonal=1).to(self.device)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        a = a + mask
        a = F.softmax(a, dim=3)
        a = torch.matmul(a, v).transpose(1, 2)
        return x + a.reshape(a.shape[0], a.shape[1], -1)


class Encoder(nn.Module):
    def __init__(self, param, device):
        """
        n: nb of Multi Head Attention
        h: nb of heads per Multi Head
        d: input dimension
        """
        super().__init__()
        n, h       = param['dim']
        k1, k2     = param['emb']
        d          = param['max_ac']*param['state_dim']
        self.n     = n
        self.emb1  = nn.Linear(d, k1)
        self.emb2  = nn.Linear(k1, k2)
        self.pos   = get_positional_encoding(k2, param['T_in'], device)
        self.multi = nn.ModuleList()
        self.fc    = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for i in range(n-1):
            if i != 0:
                self.norm1.append(nn.LayerNorm(k2))
            self.multi.append(AttentionHead(h, k2, device))
            self.norm2.append(nn.LayerNorm(k2))
            self.fc.append(nn.Linear(k2, k2))
        self.norm1.append(nn.LayerNorm(k2))
        self.multi.append(FinalHead(h, k2))

    def forward(self, x):
        x = F.relu(self.emb1(x))
        x = self.emb2(x)
        x = x + self.pos
        for i in range(self.n-1):
            # LayerNorms are before the layers
            if i != 0:
                x = self.norm1[i-1](x)
            x = self.multi[i](x)
            x = self.norm2[i](x)
            # Linear layer with skip connection
            x = x + F.relu(self.fc[i](x))
        x = self.norm1[self.n-2](x)
        x = self.multi[self.n-1](x)
        return x


class Decoder(nn.Module):
    def __init__(self, param, device):
        """
        N: nb of multi-heads
        h: nb of heads per multi-head
        n: dimension of query and key
        l: dimension of value
        """
        super().__init__()
        n, h          = param['dim']
        k1, k2        = param['emb']
        d             = param['nb_lon']*param['nb_lat']
        self.nb_class = param['nb_classes']
        self.n        = n
        self.device   = device
        self.emb1     = nn.Linear(d, k1)
        self.emb2     = nn.Linear(k1, k2)
        self.pos      = get_positional_encoding(k2, param['T_out']+1, device)
        self.multi1   = nn.ModuleList()
        self.multi2   = nn.ModuleList()
        self.fc       = nn.ModuleList()
        self.norm1    = nn.ModuleList()
        self.norm2    = nn.ModuleList()
        self.norm3    = nn.ModuleList()
        for i in range(n):
            if i != 0:
                self.norm1.append(nn.LayerNorm(k2))
            self.multi1.append(AttentionHead(h, k2, device))
            self.norm2.append(nn.LayerNorm(k2))
            self.multi2.append(DecoderHead(h, k2, device))
            self.norm3.append(nn.LayerNorm(k2))
            self.fc.append(nn.Linear(k2, k2))
        self.fc.append(nn.Linear(k2, d))
        self.cutoff = nn.parameter.Parameter(data=torch.zeros(1, self.nb_class-1))
        nn.init.xavier_uniform_(self.cutoff)

    def forward(self, x, k, v):
        x = x.float()
        x = F.relu(self.emb1(x))
        x = self.emb2(x)
        x = x + self.pos
        for i in range(self.n):
            # LayerNorms are before the layers
            if i != 0:
                x = self.norm1[i-1](x)
            x = self.multi1[i](x, masking=True)
            x = self.norm2[i](x)
            x = self.multi2[i](x, k, v)
            x = self.norm3[i](x)
            # Linear layer with skip connection
            x = F.relu(self.fc[i](x)) + x
        x        = self.fc[-1](x)
        probs    = torch.zeros(*x.shape, self.nb_class).to(self.device)
        b        = torch.zeros(*self.cutoff.shape).to(self.device)
        b[0,  0] = self.cutoff[0, 0]
        b[0, 1:] = self.cutoff[0, 1:]**2
        b[:]     = b.cumsum(dim=1)
        probs[..., 0]  = torch.sigmoid(b[0, 0] - x)
        probs[..., -1] = 1 - torch.sigmoid(b[0, -1] - x)
        for i in range(1, self.nb_class-1):
            probs[..., i] = torch.sigmoid(b[0, i] - x) - torch.sigmoid(b[0, i-1] - x)
        return probs


class TransformerED(nn.Module):
    def __init__(self, param, device):
        super().__init__()
        self.l        = param['T_out']
        self.d        = param['nb_lon']*param['nb_lat']
        self.nb_class = param['nb_classes']
        self.device   = device
        self.encoder  = Encoder(param, device)
        self.decoder  = Decoder(param, device)

    def forward(self, x, y):
        k, v = self.encoder(x)
        return self.decoder(y, k, v)[:, :-1, ...]

    def inference(self, x):
        k, v = self.encoder(x)
        pred = torch.ones(x.shape[0], self.l+1, self.d).to(self.device)
        prob = torch.ones(x.shape[0], self.l, self.d, self.nb_class).to(self.device)
        for t in range(self.l):
            y = self.decoder(pred, k, v)
            prob[:,   t, :] = y[:, t, ...]
            pred[:, t+1, :] = y[:, t, ...].argmax(dim=2)
        return pred[:, 1:, :], prob


def main():
    a = torch.ones(2, 2, 3, 3)
    print(a.triu(diagonal=1))


if __name__ == '__main__':
    main()
