import yaml
import argparse
import torch
import torch.optim as optim
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


def compute_probs(logits, device):
    probs                = torch.zeros(*logits.shape).to(device)
    probs[:, :,  0, ...] = torch.sigmoid(logits[:, :, 1, ...] - logits[:, :, 0, ...])
    probs[:, :, -1, ...] = 1 - torch.sigmoid(logits[:, :, -1, ...] - logits[:, :, 0, ...])
    for i in range(1, logits.shape[2]-1):
        probs[:, :, i, ...] = torch.sigmoid(logits[:, :, i+1, ...] - logits[:, :, 0, ...])\
                              - torch.sigmoid(logits[:, :, i, ...] - logits[:, :, 0, ...])
    return probs


def compute_cutoff(pred, device):
    cutoff = torch.zeros(pred.shape[0], pred.shape[1], pred.shape[2]+1, pred.shape[3], pred.shape[4]).to(device)
    cutoff[:, :, 1:-1, ...] = pred[:, :, 1:, ...]
    cutoff[:, :,    0, ...] = -float('inf')
    cutoff[:, :,   -1, ...] =  float('inf')
    return cutoff


def oce(pred, target, device):
    """
    Ordinal Cross-Entropy
    :param target: batch_size x T_out x nb_lon x nb_lat
    :param pred: batch_size x T_out x nb_lon x nb_lat
    :param device:
    :return: loss
    """
    cutoff = compute_cutoff(pred, device)
    temp   = target.unsqueeze(2).expand(pred.shape[0], pred.shape[1], pred.shape[2]+1, pred.shape[3], pred.shape[4])
    probs  = torch.sigmoid(torch.take_along_dim(cutoff, temp+1, dim=2)[:, :, 0, ...] - pred[:, :, 0, ...])\
             - torch.sigmoid(torch.take_along_dim(cutoff, temp, dim=2)[:, :, 0, ...] - pred[:, :, 0, ...])
    probs  = probs.clamp(1e-6, 1-1e-6)
    # If the target is NOT 0 (i.e., congestion > 0), then the loss is multiplied by 100
    temp   = torch.logical_not(torch.eq(target, torch.zeros_like(target))).float()*1e2 + 1
    loss   = -torch.log(probs)*temp
    loss   = loss.mean()
    return loss


def initialize(param, device, train=True):
    # Create model
    model = CongCNN(param)

    # Load datasets and create data loaders
    if train:
        trainset = TsagiSet(param, train=True)
        trainloader = DataLoader(trainset, batch_size=param['batch_size'], shuffle=True, pin_memory=True, num_workers=1)
    else:
        trainloader = None
    testset  = TsagiSet(param, train=False)
    testloader  = DataLoader(testset, batch_size=param['batch_size'], shuffle=False, pin_memory=True, num_workers=1)

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


def main():
    tensor = torch.rand(3, 2)
    index  = torch.tensor([[1, 0],
                           [0, 0],
                           [1, 1]])
    print(f'tensor: shape={tensor.shape}\n{tensor}')
    print(f'index: shape={index.shape}\n{index}')
    print(f'take: {torch.take_along_dim(tensor, index, dim=1)}')


if __name__ == '__main__':
    main()
