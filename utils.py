import yaml
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import TsagiSet
from models import CongCNN


def load_yaml(file_name=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    if args.config:
        yaml_file = f'config/{args.config}.yml'
    elif file_name:
        yaml_file = f'config/{file_name}.yml'
    else:
        raise RuntimeError
    with open(yaml_file) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)
    return param


def compute_probs(logits, b, device):
    probs          = torch.zeros(logits.shape[0], logits.shape[1], logits.shape[2], b.shape[1]+1).to(device)
    b              = b.unsqueeze(-1).unsqueeze(-1)
    probs[..., 0]  = F.sigmoid(b[:, 0] - logits)
    probs[..., -1] = 1 - F.sigmoid(b[:, -1] - logits)
    for i in range(1, b.shape[1]):
        probs[..., i] = F.sigmoid(b[:, i] - logits) - F.sigmoid(b[:, i-1] - logits)
    return probs


def compute_cutoff(b, device):
    cutoff                  = torch.zeros(b.shape[0], b.shape[1]+2).to(device)
    cutoff[:, 1:-1]         = b[:]
    cutoff[:, 0]            = -float('inf')
    cutoff[:, b.shape[1]+1] = float('inf')
    return cutoff


def oce(pred, b, target, device):
    """
    Ordinal Cross-Entropy
    :param target: batch_size x T_out x nb_lon x nb_lat
    :param pred: batch_size x T_out x nb_lon x nb_lat
    :param b: batch_size x (nb_classes - 1)
    :param device:
    :return: loss
    """
    cutoff = compute_cutoff(b, device)
    probs  = F.sigmoid(torch.take(cutoff, target+1) - pred) - F.sigmoid(torch.take(cutoff, target) - pred)
    probs  = probs.clamp(1e-6, 1-1e-6)
    loss   = -torch.log(probs)
    loss   = loss.mean()
    return loss


def initialize(param, device):
    # Create model
    model = CongCNN(param)

    # Load datasets
    trainset = TsagiSet(param, train=True)
    testset  = TsagiSet(param, train=False)

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=param['batch_size'], shuffle=True, pin_memory=True, num_workers=1)
    testloader  = DataLoader(testset, batch_size=param['batch_size'], shuffle=True, pin_memory=True, num_workers=1)

    # Load model
    if param['load']:
        checkpoint = torch.load(f'models/{param["name"]}/{param["model"]}', map_location='cpu')
        model.load_state_dict(checkpoint)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    model.to(device)

    # Set optimizer
    if param['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    elif param['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    else:
        raise NotImplementedError

    return trainloader, testloader, model, optimizer
