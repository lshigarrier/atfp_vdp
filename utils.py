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


def compute_probs(logits, b):
    probs = torch.zeros(logits.shape[0], logits.shape[1], logits.shape[2], b.shape[1]+1)
    cutoff = compute_cutoff(b).unsqueeze(-1).unsqueeze(-1)
    for i in range(cutoff.shape[1]-1):
        probs[..., i] = F.sigmoid(cutoff[:, i+1] - logits) - F.sigmoid(cutoff[:, i] - logits)
    return probs


def compute_cutoff(b):
    cutoff = torch.zeros(b.shape[0], b.shape[1]+2)
    cutoff[:, 0] = -float('inf')
    cutoff[:, b.shape[1]+1] = float('inf')
    cutoff[:, 1] = b[:, 0]
    for i in range(2, b.shape[1]+1):
        cutoff[:, i] = cutoff[:, i-1] + b[:, i-1]
    return cutoff


def oce(pred, b, target):
    """
    Ordinal Cross-Entropy
    :param target: batch_size x T_out x nb_lon x nb_lat
    :param pred: batch_size x T_out x nb_lon x nb_lat
    :param b: batch_size x (nb_classes - 1)
    :return: loss
    """
    cutoff = compute_cutoff(b)
    loss   = -torch.log(F.sigmoid(torch.take(cutoff, target+1) - pred) - F.sigmoid(torch.take(cutoff, target) - pred))
    return loss.mean()


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
