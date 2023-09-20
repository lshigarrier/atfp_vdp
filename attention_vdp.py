import math
import torch
import torch.nn as nn
from attention import get_positional_encoding
from vdp import quadratic_vdp, relu_vdp, sigmoid_vdp, softmax_vdp, residual_vdp, LinearVDP, LayerNormVDP


def patchify(images, device, n_patches=7):
    n, c, h, w = images.shape

    # Patchify method is implemented for square images only
    assert h == w

    patches    = torch.zeros(n, n_patches**2, h*w*c//n_patches**2).to(device)
    patch_size = h//n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i*patch_size: (i + 1)*patch_size, j*patch_size: (j + 1)*patch_size]
                patches[idx, i*n_patches + j] = patch.flatten()
    return patches


class AttentionHeadVDP(nn.Module):
    """
    Multi Head Attention with skip connection
    h: nb of heads
    d: input dimension
    """
    def __init__(self, h, d, device, mode='identity', var_init=1e-8, tol=1e-3):
        super().__init__()
        if d % h != 0:
            raise RuntimeError
        self.rd     = math.sqrt(d)
        self.h      = h
        self.device = device
        self.mode   = mode
        self.tol    = tol
        self.query  = LinearVDP(d, h*(d//h), bias=False, var_init=var_init, tol=tol)
        self.key    = LinearVDP(d, h*(d//h), bias=False, var_init=var_init, tol=tol)
        self.value  = LinearVDP(d, h*(d//h), bias=False, var_init=var_init, tol=tol)

    def forward(self, x, var_x, masking=False):
        q, var_q = self.query(x, var_x)  # b x l x h.s where s = d//h
        q = q.reshape(q.shape[0], q.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        var_q = var_q.reshape(var_q.shape[0], var_q.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        k, var_k = self.key(x, var_x)  # b x l x h.s
        k = k.reshape(k.shape[0], k.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        var_k = var_k.reshape(var_k.shape[0], var_k.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        v, var_v = self.value(x, var_x)  # b x l x h.s
        v = v.reshape(v.shape[0], v.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        var_v = var_v.reshape(var_v.shape[0], var_v.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        a, var_a = quadratic_vdp(q, var_q, k.transpose(2, 3), var_k.transpose(2, 3), tol=self.tol)  # b x h x l x l
        a, var_a = a/self.rd, var_a/self.rd**2
        if masking:
            mask = torch.ones(*a.shape).triu(diagonal=1).to(self.device)
            mask = mask.masked_fill(mask==1, float('-inf'))
            a    = a + mask
        a, var_a = softmax_vdp(a, var_a, tol=self.tol)  # b x h x l x l
        a, var_a = quadratic_vdp(a, var_a, v, var_v, tol=self.tol)
        a, var_a = a.transpose(1, 2), var_a.transpose(1, 2)  # b x l x h x s
        return residual_vdp(x, var_x, a.reshape(a.shape[0], a.shape[1], -1),
                            var_a.reshape(var_a.shape[0], var_a.shape[1], -1),
                            mode=self.mode, tol=self.tol) # b x l x h.s


class FinalHeadVDP(nn.Module):
    """
    Last Multi Head of the encoder
    """
    def __init__(self, h, d, var_init=1e-8, tol=1e-3):
        super().__init__()
        if d % h != 0:
            raise RuntimeError
        self.h     = h
        self.key   = LinearVDP(d, h*(d//h), bias=False, var_init=var_init, tol=tol)
        self.value = LinearVDP(d, h*(d//h), bias=False, var_init=var_init, tol=tol)

    def forward(self, x, var_x):
        k, var_k = self.key(x, var_x)
        k = k.reshape(k.shape[0], k.shape[1], self.h, -1).transpose(1, 2)
        var_k = var_k.reshape(var_k.shape[0], var_k.shape[1], self.h, -1).transpose(1, 2)
        v, var_v = self.value(x, var_x)
        v = v.reshape(v.shape[0], v.shape[1], self.h, -1).transpose(1, 2)
        var_v = var_v.reshape(var_v.shape[0], var_v.shape[1], self.h, -1).transpose(1, 2)
        return k, var_k, v, var_v


class DecoderHeadVDP(nn.Module):
    """
    Multi Head Attention using key and value from the encoder
    """
    def __init__(self, h, d, device, mode='identity', var_init=1e-8, tol=1e-3):
        super().__init__()
        if d % h != 0:
            raise RuntimeError
        self.rd     = math.sqrt(d)
        self.h      = h
        self.device = device
        self.mode   = mode
        self.tol    = tol
        self.query  = LinearVDP(d, h*(d//h), bias=False, var_init=var_init, tol=tol)

    def forward(self, x, var_x, k, var_k, v, var_v):
        q, var_q = self.query(x, var_x)
        q = q.reshape(q.shape[0], q.shape[1], self.h, -1).transpose(1, 2)
        var_q = var_q.reshape(var_q.shape[0], var_q.shape[1], self.h, -1).transpose(1, 2)
        a, var_a = quadratic_vdp(q, var_q, k.transpose(2, 3), var_k.transpose(2, 3), tol=self.tol)
        a, var_a = a/self.rd, var_a/self.rd**2
        mask = torch.ones(*a.shape).triu(diagonal=1).to(self.device)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        a = a + mask
        a, var_a = softmax_vdp(a, var_a, tol=self.tol)
        a, var_a = quadratic_vdp(a, var_a, v, var_v, tol=self.tol)
        a, var_a = a.transpose(1, 2), var_a.transpose(1, 2)
        return residual_vdp(x, var_x, a.reshape(a.shape[0], a.shape[1], -1),
                            var_a.reshape(var_a.shape[0], var_a.shape[1], -1),
                            mode=self.mode, tol=self.tol)

class EncoderVDP(nn.Module):
    def __init__(self, param, device):
        """
        n: nb of Multi Head Attention
        h: nb of heads per Multi Head
        d: input dimension
        """
        super().__init__()
        n, h       = param['dim']
        emb_dim    = param['emb']
        k          = emb_dim[-1]
        d          = param['max_ac']*param['state_dim']
        self.n     = n
        self.q     = len(emb_dim)
        self.mode  = param['residual']
        self.tol   = param['tol']
        self.pos   = get_positional_encoding(k, param['T_in'], device)
        self.emb   = nn.ModuleList()
        self.multi = nn.ModuleList()
        self.fc    = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.emb.append(LinearVDP(d, emb_dim[0], var_init=param['var_init'], tol=self.tol))
        for i in range(len(emb_dim)-1):
            self.emb.append(LinearVDP(emb_dim[i], emb_dim[i+1], var_init=param['var_init'], tol=self.tol))
        for i in range(n-1):
            if i != 0:
                 self.norm1.append(LayerNormVDP(k, var_init=param['var_init'], tol=param['tol']))
            self.multi.append(AttentionHeadVDP(h, k, device, self.mode, param['var_init'], self.tol))
            self.norm2.append(LayerNormVDP(k, var_init=param['var_init'], tol=self.tol))
            self.fc.append(LinearVDP(k, k, var_init=param['var_init'], tol=self.tol))
        self.norm1.append(LayerNormVDP(k, var_init=param['var_init'], tol=self.tol))
        self.multi.append(FinalHeadVDP(h, k, param['var_init'], self.tol))

    def forward(self, x):
        var_x = torch.zeros_like(x)
        for i in range(self.q-1):
            x, var_x = relu_vdp(*self.emb[i](x, var_x), tol=self.tol)
        x, var_x = self.emb[-1](x, var_x)
        x = x + self.pos
        for i in range(self.n-1):
            # LayerNorms are before the layers
            if i != 0:
                x, var_x = self.norm1[i-1](x, var_x)
            x, var_x = self.multi[i](x, var_x)
            x, var_x = self.norm2[i](x, var_x)
            # Linear layer with skip connection
            x0, var_x0 = x[:], var_x[:]
            x, var_x, j_relu = relu_vdp(*self.fc[i](x, var_x), return_jac=True, tol=self.tol)
            # torch.matmul(j_relu, self.fc[i].get_jac())
            x, var_x = residual_vdp(x0, var_x0, x, var_x, mode=self.mode, tol=self.tol)
        x, var_x = self.norm1[self.n-2](x, var_x)
        k, var_k, v, var_v = self.multi[self.n-1](x, var_x)
        return k, var_k, v, var_v


class DecoderVDP(nn.Module):
    def __init__(self, param, device):
        """
        N: nb of multi-heads
        h: nb of heads per multi-head
        n: dimension of query and key
        l: dimension of value
        """
        super().__init__()
        n, h          = param['dim']
        emb_dim       = param['emb']
        k             = emb_dim[-1]
        if param['predict_spot']:
            d = 1
        else:
            d = param['nb_lon']*param['nb_lat']
        self.nb_class = param['nb_classes']
        self.n        = n
        self.q        = len(emb_dim)
        self.device   = device
        self.mode     = param['residual']
        self.tol      = param['tol']
        self.pos      = get_positional_encoding(k, param['T_out']+1, device)
        self.emb      = nn.ModuleList()
        self.multi1   = nn.ModuleList()
        self.multi2   = nn.ModuleList()
        self.fc       = nn.ModuleList()
        self.norm1    = nn.ModuleList()
        self.norm2    = nn.ModuleList()
        self.norm3    = nn.ModuleList()
        self.emb.append(LinearVDP(d, emb_dim[0], var_init=param['var_init'], tol=self.tol))
        for i in range(len(emb_dim)-1):
            self.emb.append(LinearVDP(emb_dim[i], emb_dim[i+1], var_init=param['var_init'], tol=self.tol))
        for i in range(n):
            if i != 0:
                self.norm1.append(LayerNormVDP(k, var_init=param['var_init'], tol=self.tol))
            self.multi1.append(AttentionHeadVDP(h, k, device, self.mode, param['var_init'], self.tol))
            self.norm2.append(LayerNormVDP(k, var_init=param['var_init'], tol=self.tol))
            self.multi2.append(DecoderHeadVDP(h, k, device, self.mode, param['var_init'], self.tol))
            self.norm3.append(LayerNormVDP(k, var_init=param['var_init'], tol=self.tol))
            self.fc.append(LinearVDP(k, k, var_init=param['var_init'], tol=self.tol))
        self.fc.append(LinearVDP(k, d, var_init=param['var_init'], tol=self.tol))
        self.cutoff = nn.parameter.Parameter(data=torch.zeros(1, self.nb_class-1))
        nn.init.xavier_uniform_(self.cutoff)

    def forward(self, x, k, var_k, v, var_v):
        x = x.float()
        var_x = torch.zeros_like(x)
        for i in range(self.q-1):
            x, var_x = relu_vdp(*self.emb[i](x, var_x), tol=self.tol)
        x, var_x = self.emb[-1](x, var_x)
        x = x + self.pos
        for i in range(self.n):
            # LayerNorms are before the layers
            if i != 0:
                x, var_x = self.norm1[i-1](x, var_x)
            x, var_x = self.multi1[i](x, var_x, masking=True)
            x, var_x = self.norm2[i](x, var_x)
            x, var_x = self.multi2[i](x, var_x, k, var_k, v, var_v)
            x, var_x = self.norm3[i](x, var_x)
            # Linear layer with skip connection
            x0, var_x0 = x[:], var_x[:]
            x, var_x, j_relu = relu_vdp(*self.fc[i](x, var_x), return_jac=True, tol=self.tol)
            x, var_x = residual_vdp(x0, var_x0, x, var_x, mode=self.mode, tol=self.tol)
        x, var_x = self.fc[-1](x, var_x)
        probs    = torch.zeros(*x.shape, self.nb_class).to(self.device)
        var_prob = torch.zeros(*x.shape, self.nb_class).to(self.device)
        b        = torch.zeros_like(self.cutoff).to(self.device)
        b[0,  0] = self.cutoff[0, 0]
        b[0, 1:] = self.cutoff[0, 1:]**2
        b[:]     = b.cumsum(dim=1)
        p0       = b[0, 0] - x
        probs[..., 0], var_prob[..., 0] = sigmoid_vdp(p0, var_x)
        p1       = b[0, -1] - x
        probs[..., -1], var_prob[..., -1] = sigmoid_vdp(p1, var_x)
        probs[..., -1] = 1 - probs[..., -1]
        for i in range(1, self.nb_class-1):
            pi               = b[0, i] - x
            pi_1             = b[0, i-1] - x
            xi, var_i        = sigmoid_vdp(pi, var_x)
            xi_1, var_i_1    = sigmoid_vdp(pi_1, var_x)
            probs[..., i]    = xi - xi_1
            var_prob[..., i] = var_i + var_i_1
        return probs, var_prob


class TransformerED_VDP(nn.Module):
    def __init__(self, param, device):
        super().__init__()
        self.l        = param['T_out']
        if param['predict_spot']:
            self.d = 1
        else:
            self.d = param['nb_lon']*param['nb_lat']
        self.nb_class = param['nb_classes']
        self.device   = device
        self.encoder  = EncoderVDP(param, device)
        self.decoder  = DecoderVDP(param, device)

    def forward(self, x, y):
        k, var_k, v, var_v = self.encoder(x)
        probs, var_prob    = self.decoder(y, k, var_k, v, var_v)
        return probs[:, :-1, ...], var_prob[:, :-1, ...]

    def inference(self, x):
        k, var_k, v, var_v = self.encoder(x)
        pred     = -2*torch.ones(x.shape[0], self.l+1, self.d).to(self.device)
        prob     = torch.ones(x.shape[0], self.l, self.d, self.nb_class).to(self.device)
        var_prob = torch.ones(x.shape[0], self.l, self.d, self.nb_class).to(self.device)
        for t in range(self.l):
            y, var_y = self.decoder(pred, k, var_k, v, var_v)
            prob[:,   t, :]   = y[:, t, ...]
            var_prob[:, t, :] = var_y[:, t, ...]
            pred[:, t+1, :]   = y[:, t, ...].argmax(dim=2)
        return pred[:, 1:, :], prob, var_prob


class ViT_VDP(nn.Module):
    def __init__(self, param, device):
        super().__init__()
        """
        n: nb of Multi Head Attention
        h: nb of heads per Multi Head
        d: input dimension
        """
        super().__init__()
        n, h        = param['dim']
        emb_dim     = param['emb']
        k           = emb_dim[0]
        d           = param['patch_size']**2
        self.n      = n
        self.q      = len(emb_dim)
        self.device = device
        self.nb_p   = param['n_patches']
        self.mode   = param['residual']
        self.tol    = param['tol']
        self.pos    = get_positional_encoding(k, param['n_patches']**2, device)
        self.clas   = nn.ModuleList()
        self.multi  = nn.ModuleList()
        self.fc     = nn.ModuleList()
        self.norm1  = nn.ModuleList()
        self.norm2  = nn.ModuleList()
        self.emb    = LinearVDP(d, k, var_init=param['var_init'], tol=self.tol)
        for i in range(n):
            if i != 0:
                self.norm1.append(LayerNormVDP(k, var_init=param['var_init'], tol=param['tol']))
            self.multi.append(AttentionHeadVDP(h, k, device, self.mode, param['var_init'], self.tol))
            self.norm2.append(LayerNormVDP(k, var_init=param['var_init'], tol=self.tol))
            self.fc.append(LinearVDP(k, k, var_init=param['var_init'], tol=self.tol))
        for i in range(len(emb_dim)-1):
            self.clas.append(LinearVDP(emb_dim[i], emb_dim[i+1], var_init=param['var_init'], tol=self.tol))
        self.classifier = LinearVDP(emb_dim[-1], param['nb_classes'], var_init=param['var_init'], tol=self.tol)

    def forward(self, x, _):
        x        = patchify(x, self.device, self.nb_p)
        var_x    = torch.zeros_like(x)
        x, var_x = self.emb(x, var_x)
        x        = x + self.pos
        for i in range(self.n-1):
            # LayerNorms are before the layers
            if i != 0:
                x, var_x = self.norm1[i-1](x, var_x)
            x, var_x = self.multi[i](x, var_x)
            x, var_x = self.norm2[i](x, var_x)
            # Linear layer with skip connection
            x0, var_x0 = x[:], var_x[:]
            x, var_x, j_relu = relu_vdp(*self.fc[i](x, var_x), return_jac=True, tol=self.tol)
            # torch.matmul(j_relu, self.fc[i].get_jac())
            x, var_x = residual_vdp(x0, var_x0, x, var_x, mode=self.mode, tol=self.tol)
        # Global average pooling
        x     = x.mean(dim=1)
        var_x = var_x.mean(dim=1)/self.nb_p**2
        # Classifier
        for i in range(self.q-1):
            x, var_x = relu_vdp(*self.clas[i](x, var_x), tol=self.tol)
        x, var_x = softmax_vdp(*self.classifier(x, var_x), tol=self.tol)
        return x, var_x

    def inference(self, x):
        x, var_x = self.forward(x, x)
        return x.argmax(dim=-1), x, var_x


def main():
    index = torch.tensor([[1, 2, 3], [3, 1, 2]])
    value = torch.tensor([0.1, 0.2, 0.3])
    print(value[index])


if __name__ == '__main__':
    main()
