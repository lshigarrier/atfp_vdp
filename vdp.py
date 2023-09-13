import math
# import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_vdp(probs, var_prob, target, model, param, device):
    """
    Ordinal Cross-Entropy
    :param probs: batch_size x T_out x (nb_lon x nb_lat) x nb_classes
    :param var_prob: batch_size x T_out x (nb_lon x nb_lat) x nb_classes
    :param target: batch_size x T_out x (nb_lon x nb_lat)
    :param model:
    :param param:
    :param device:
    :return: loss
    """
    # Compute the regularization term
    kl = 0
    for name, p in model.named_parameters():
        if p.requires_grad and 'cutoff' not in name:
            if 'sigma' in name:
                sig2 = F.softplus(p)
                kl  += torch.nan_to_num(sig2, nan=param['tol'], posinf=1/param['tol'], neginf=-1/param['tol']).sum()\
                       - torch.nan_to_num(torch.log(sig2 + param['tol']),
                                          nan=param['tol'], posinf=1/param['tol'], neginf=-1/param['tol']).sum()
            else:
                kl += torch.sum(torch.nan_to_num(p**2, nan=param['tol'], posinf=1/param['tol'], neginf=-1/param['tol']))
    # Compute the expected negative log-likelihood
    probs   = torch.nan_to_num(probs,
                               nan=param['tol'],
                               posinf=1/param['tol'],
                               neginf=-1/param['tol']).clamp(param['tol'], 1-param['tol'])
    target  = target[:, 1:, :]
    if param['balance']:
        nb_total = target.ne(-1).int().sum().item()
        coef     = torch.eq(target, 0)
    mask     = target.ne(-1).int()
    target   = mask*target
    weights  = torch.take(torch.tensor(param['weights']).to(device), target)
    target   = F.one_hot(target, num_classes=param['nb_classes'])
    p_true   = torch.matmul(target.unsqueeze(-2).float(), probs.unsqueeze(-1)).squeeze()
    var_prob = torch.clamp(torch.nan_to_num(var_prob, nan=param['tol'], posinf=1/param['tol'], neginf=-1/param['tol']),
                           min=param['tol'], max=1/param['tol'])
    # print(f'sigma: {scipy.stats.describe(var_prob.cpu().detach().flatten())}')
    inv_var  = torch.div(1, var_prob + param['tol'])
    nll      = torch.nan_to_num(mask*weights*(1 - p_true)**param['focus']*
                                torch.matmul(((target - probs)*inv_var).unsqueeze(-2),
                                            (target - probs).unsqueeze(-1)).squeeze(),
                                nan=param['tol'], posinf=1/param['tol'], neginf=-1/param['tol'])
    nll     += torch.nan_to_num(mask*weights*(1 - p_true)**param['focus']*
                                torch.log(var_prob + param['tol']).sum(dim=-1),
                                nan=param['tol'], posinf=1/param['tol'], neginf=-1/param['tol'])
    # Remove elements from the loss by multiplying them by 0
    # such that the proportion of 0 in target is 1/param['nb_classes']
    if param['balance']:
        nb_zero = coef.long().sum().item()
        remove = int(max(nb_zero - nb_total / param['nb_classes'], 0))
        if remove > 0:
            index = torch.nonzero(coef, as_tuple=False)
            index = index[torch.randperm(nb_zero)]
            index = index[:nb_zero - remove, :]
            coef[index.t().tolist()] = False
            coef = coef.masked_fill(coef, 0).masked_fill(~coef, 1)
            nll  = nll*coef
    shapes = probs.shape
    return nll.sum()/(shapes[0]*shapes[1]*shapes[2]) + param['kl_factor']*kl


def quadratic_vdp(x, var_x, y, var_y, tol=1e-3):
    return torch.matmul(x, y), torch.clamp(torch.nan_to_num(torch.matmul(var_x + x**2, var_y) + torch.matmul(var_x, y**2),
                                                nan=tol, posinf=1/tol, neginf=-1/tol), min=tol, max=1/tol)


def quadratic_jac(x, jac_x, y, jac_y):
    return torch.matmul(jac_x, y) + torch.matmul(x, jac_y)


def relu_vdp(x, var_x, return_jac=False, tol=1e-3):
    x   = F.relu(x)
    der = torch.logical_not(torch.eq(x, 0)).long()
    if return_jac:
        return x, torch.clamp(torch.nan_to_num(var_x*der, nan=tol, posinf=1/tol, neginf=-1/tol), min=tol, max=1/tol), der
    else:
        return x, torch.clamp(torch.nan_to_num(var_x*der, nan=tol, posinf=1/tol, neginf=-1/tol), min=tol, max=1/tol)


def sigmoid_vdp(x, var_x, tol=1e-3):
    x   = torch.sigmoid(x)
    der = x*(1 - x)
    return x, torch.clamp(torch.nan_to_num(var_x*der**2, nan=tol, posinf=1/tol, neginf=-1/tol), min=tol, max=1/tol)


def softmax_vdp(x, var_x, return_jac=False, tol=1e-3):
    """
    To avoid an out-of-memory error, we must neglect the off-diagonal terms of the Jacobian
    """
    prob = F.softmax(x, dim=-1)
    der  = prob*(1 - prob)
    if return_jac:
        return prob, torch.clamp(torch.nan_to_num(var_x*der**2, nan=tol, posinf=1/tol, neginf=-1/tol), min=tol, max=1/tol), der
    else:
        return prob, torch.clamp(torch.nan_to_num(var_x*der**2, nan=tol, posinf=1/tol, neginf=-1/tol), min=tol, max=1/tol)
    # jac = torch.diag_embed(prob) - torch.matmul(prob.unsqueeze(-1), prob.unsqueeze(-2))
    # var = torch.matmul(jac*var_x.unsqueeze(-2), jac.transpose(-1, -2))
    # return prob, torch.diagonal(var, dim1=-2, dim2=-1)


def residual_vdp(x, var_x, f, var_f=None, jac=None, mode='taylor', tol=1e-3):
    if mode == 'taylor':
        if (var_f is None) or (jac is None):
            raise RuntimeError
        return x + f, torch.clamp(torch.nan_to_num(torch.maximum(var_x + var_f + 2*(jac*(var_x + x**2) - x*f), torch.tensor(0)),
                                                   nan=tol, posinf=1/tol, neginf=-1/tol),
                                  min=tol, max=1/tol)
    elif mode == 'independence':
        if var_f is None:
            raise RuntimeError
        return x + f, torch.clamp(torch.nan_to_num(var_x + var_f, nan=tol, posinf=1/tol, neginf=-1/tol),
                                  min=tol, max=1/tol)
    elif mode == 'identity':
        return x + f, torch.clamp(torch.nan_to_num(var_x, nan=tol, posinf=-1/tol, neginf=-1/tol),
                                  min=tol, max=1/tol)
    else:
        raise NotImplementedError


class LinearVDP(nn.Module):
    def __init__(self, in_features, out_features, bias=True, tol=1e-3):
        super().__init__()
        self.size_in, self.size_out, self.bias, self.tol = in_features, out_features, bias, tol
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
            bound = 1/math.sqrt(in_features)
            nn.init.uniform_(self.b, -bound, bound)
            nn.init.uniform_(self.b_sig, -bound, bound)

    def forward(self, mu, sigma):
        sig2  = F.softplus(self.sigma)
        mu    = mu.transpose(-1, -2)
        sigma = sigma.transpose(-1, -2)
        mean  = torch.matmul(self.weights, mu)
        var   = torch.matmul(sig2 + self.weights**2, sigma) + torch.matmul(sig2, mu**2)
        if self.bias:
            return mean.transpose(-2, -1) + self.b, torch.clamp(torch.nan_to_num(var.transpose(-2, -1) + F.softplus(self.b_sig),
                                                                                 nan=self.tol, posinf=1/self.tol, neginf=-1/self.tol),
                                                                min=self.tol, max=1/self.tol)
        else:
            return mean.transpose(-2, -1), torch.clamp(torch.nan_to_num(var.transpose(-2, -1),
                                                                        nan=self.tol, posinf=1/self.tol, neginf=-1/self.tol),
                                                       min=self.tol, max=1/self.tol)

    def get_jac(self):
        return self.weights.transpose(-2, -1)


class LayerNormVDP(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True, tol=1e-3):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.tol = tol
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
            return (mu - mean)/torch.sqrt(var + self.tol)*self.weights + self.b,\
                   torch.clamp(torch.nan_to_num(sigma/(var + self.tol)*F.softplus(self.sigma) + F.softplus(self.b_sig),
                                                nan=self.tol, posinf=1/self.tol, neginf=-1/self.tol),
                               min=self.tol, max=1/self.tol)
        else:
            return (mu - mean)/torch.sqrt(var + self.tol),\
                   torch.clamp(torch.nan_to_num(sigma/(var + self.tol),
                                                nan=self.tol, posinf=1/self.tol, neginf=-1/self.tol),
                               min=self.tol, max=1/self.tol)
