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
                kl  += sig2.sum() - torch.log(sig2 + param['tol']).sum()
            else:
                kl += torch.sum(p**2)
    # Compute the expected negative log-likelihood
    probs   = probs.clamp(param['tol'], 1-param['tol'])
    target  = F.one_hot(target[:, 1:, :], num_classes=param['nb_classes'])
    inv_var = torch.div(1, var_prob)
    nll     = torch.matmul(((target - probs)*inv_var).unsqueeze(-2), (target - probs).unsqueeze(-1)).sum()
    nll    += torch.log(var_prob.prod(dim=-1) + param['tol']).sum()
    return nll + kl


def linear_vdp(x, var_x, y, var_y):
    return torch.matmul(x, y), torch.matmul(var_x + x**2, var_y) + torch.matmul(var_x, y**2)


def relu_vdp(x, var_x):
    x   = F.relu(x)
    der = torch.logical_not(torch.eq(x, 0)).long()
    return x, var_x*der


def sigmoid_vdp(x, var_x):
    x   = torch.sigmoid(x)
    der = x*(1-x)
    return x, var_x*der**2


def softmax_vdp(x, var_x):
    prob = F.softmax(x, dim=-1)
    jac = torch.diag_embed(prob) - torch.matmul(prob.unsqueeze(-1), prob.unsqueeze(-2))
    var = torch.matmul(jac * var_x.unsqueeze(-2), jac.transpose(-1, -2))
    return prob, torch.diagonal(var, dim1=-2, dim2=-1)


class LinearVDP(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.size_in, self.size_out, self.bias = in_features, out_features, bias
        weights  = torch.zeros(out_features, in_features)
        variance = torch.zeros(out_features, in_features)
        self.weights   = nn.Parameter(weights)
        self.sigma  = nn.Parameter(variance)
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.sigma)
        if bias:
            b = torch.zeros(out_features)
            b_var = torch.zeros(out_features)
            self.b = nn.Parameter(b)
            self.b_var = nn.Parameter(b_var)
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.b, -bound, bound)
            nn.init.uniform_(self.b_var, -bound, bound)

    def forward(self, mu, sigma):
        sig2 = self.sigma**2
        if self.bias:
            mean = torch.matmul(self.weights, mu) + self.b
            var = torch.matmul(sig2  + self.weights ** 2, sigma) + torch.matmul(sig2, mu ** 2) + self.b_var
        else:
            mean = torch.matmul(self.weights, mu)
            var  = torch.matmul(sig2  + self.weights**2, sigma) + torch.matmul(sig2, mu**2)
        return mean, var
