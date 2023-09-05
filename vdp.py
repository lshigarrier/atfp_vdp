import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_vdp(probs, var_prob, target, model, param):
    """
    Ordinal Cross-Entropy
    :param probs: batch_size x T_out x (nb_lon x nb_lat) x nb_classes
    :param var_prob: batch_size x T_out x (nb_lon x nb_lat) x nb_classes
    :param target: batch_size x T_out x (nb_lon x nb_lat)
    :param model:
    :param param:
    :return: loss
    """
    # Compute the regularization term
    kl = 0
    for name, p in model.named_parameters():
        if p.requires_grad and 'cutoff' not in name:
            if 'sigma' in name:
                sig2 = p**2
                kl  += torch.nan_to_num(sig2, nan=param['tol'], posinf=param['tol'], neginf=param['tol']).sum()\
                       - torch.nan_to_num(torch.log(sig2 + param['tol']),
                                          nan=param['tol'], posinf=param['tol'], neginf=param['tol']).sum()
            else:
                kl += torch.sum(torch.nan_to_num(p**2, nan=param['tol'], posinf=param['tol'], neginf=param['tol']))
    # Compute the expected negative log-likelihood
    probs   = probs.clamp(param['tol'], 1-param['tol'])
    target  = F.one_hot(target[:, 1:, :], num_classes=param['nb_classes'])
    inv_var = torch.div(1, var_prob)
    nll     = torch.nan_to_num(torch.matmul(((target - probs)*inv_var).unsqueeze(-2), (target - probs).unsqueeze(-1)),
                               nan=param['tol'], posinf=param['tol'], neginf=param['tol']).sum()
    nll    += torch.nan_to_num(torch.log(var_prob.prod(dim=-1) + param['tol']),
                               nan=param['tol'], posinf=param['tol'], neginf=param['tol']).sum()
    shapes  = probs.shape
    return nll/(shapes[0]*shapes[1]*shapes[2]) + kl


def quadratic_vdp(x, var_x, y, var_y):
    return torch.matmul(x, y), torch.matmul(var_x + x**2, var_y) + torch.matmul(var_x, y**2)


def relu_vdp(x, var_x):
    x   = F.relu(x)
    der = torch.logical_not(torch.eq(x, 0)).long()
    return x, var_x*der


def sigmoid_vdp(x, var_x):
    x   = torch.sigmoid(x)
    der = x*(1 - x)
    return x, var_x*der**2


def softmax_vdp(x, var_x):
    """
    To avoid an out-of-memory error, we must neglect the off-diagonal terms of the Jacobian
    """
    prob = F.softmax(x, dim=-1)
    der  = prob*(1 - prob)
    return prob, var_x*der**2
    # jac = torch.diag_embed(prob) - torch.matmul(prob.unsqueeze(-1), prob.unsqueeze(-2))
    # var = torch.matmul(jac*var_x.unsqueeze(-2), jac.transpose(-1, -2))
    # return prob, torch.diagonal(var, dim1=-2, dim2=-1)


def residual_vdp(x, var_x, f, var_f=None, jac=None, mode='taylor'):
    if mode == 'taylor':
        if (var_f is None) or (jac is None):
            raise RuntimeError
        return x + f, torch.maximum(var_x + var_f + 2*(jac*(var_x + x**2) - x*f), torch.tensor(0))
    elif mode == 'independence':
        if var_f is None:
            raise RuntimeError
        return x + f, var_x + var_f
    elif mode == 'identity':
        return x + f, var_x
    else:
        raise NotImplementedError


class LinearVDP(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.size_in, self.size_out, self.bias = in_features, out_features, bias
        weights  = torch.zeros(out_features, in_features)
        std      = torch.zeros(out_features, in_features)
        self.weights = nn.Parameter(weights)
        self.sigma   = nn.Parameter(std)
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.sigma)
        if bias:
            b = torch.zeros(out_features)
            b_sig = torch.zeros(out_features)
            self.b = nn.Parameter(b)
            self.b_sig = nn.Parameter(b_sig)
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.b, -bound, bound)
            nn.init.uniform_(self.b_sig, -bound, bound)

    def forward(self, mu, sigma):
        sig2  = self.sigma**2
        mu    = mu.transpose(-1, -2)
        sigma = sigma.transpose(-1, -2)
        mean  = torch.matmul(self.weights, mu)
        var   = torch.matmul(sig2 + self.weights**2, sigma) + torch.matmul(sig2, mu**2)
        if self.bias:
            return mean.transpose(-2, -1) + self.b, var.transpose(-2, -1) + self.b_sig**2
        else:
            return mean.transpose(-2, -1), var.transpose(-2, -1)


class LayerNormVDP(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            weights = torch.ones(normalized_shape)
            std     = torch.zeros(normalized_shape)
            b       = torch.zeros(normalized_shape)
            b_sig   = torch.zeros(normalized_shape)
            self.weights = nn.Parameter(weights)
            self.sigma   = nn.Parameter(std)
            self.b       = nn.Parameter(b)
            self.b_sig   = nn.Parameter(b_sig)
            bound = 1 / math.sqrt(normalized_shape)
            nn.init.uniform_(self.sigma, -bound, bound)
            nn.init.uniform_(self.b_sig, -bound, bound)

    def forward(self, mu, sigma):
        mean = mu.mean(dim=-1, keepdim=True)
        var  = mu.var(dim=-1, keepdim=True)
        if self.elementwise_affine:
            return (mu - mean)/torch.sqrt(var + self.eps)*self.weights + self.b, sigma/(var + self.eps)*self.sigma**2 + self.b_sig**2
        else:
            return (mu - mean)/torch.sqrt(var + self.eps), sigma/(var + self.eps)
