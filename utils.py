import yaml
import argparse
# import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TsagiSet
from attention import TransformerED


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


def oce(probs, target, param):
    """
    Ordinal Cross-Entropy
    :param target: batch_size x T_out x (nb_lon x nb_lat)
    :param probs: batch_size x T_out x (nb_lon x nb_lat) x nb_classes
    :param param:
    :return: loss
    """
    probs  = probs.clamp(param['tol'], 1-param['tol'])
    target = target[:, 1:, :]
    idx    = target.unsqueeze(3).expand(*target.shape, param['nb_classes'])
    probs  = torch.take_along_dim(probs, idx, dim=3)[..., 0]
    loss = -torch.log(probs)
    # Remove elements from the loss by multiplying them by 0
    # such that the proportion of 0 in target is 1/param['nb_classes']
    nb_total = target.numel()
    coef     = torch.eq(target, 0)
    nb_zero  = coef.long().sum().item()
    remove   = int(max(nb_zero - nb_total/param['nb_classes'], 0))
    if remove > 0:
        index = torch.nonzero(coef, as_tuple=False)
        index = index[torch.randperm(nb_zero)]
        index = index[:nb_zero-remove,:]
        coef[index.t().tolist()] = False
        coef  = coef.masked_fill(coef, 0).masked_fill(~coef, 1)
        loss  = loss*coef
    loss = loss.mean()
    return loss


def initialize(param, device, train=True):
    # Print param
    for key in param:
        print(f'{key}: {param[key]}')
    print(f'device: {device}')
    # Create model
    print('Initialize model')
    model = TransformerED(param, device)
    nb_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {nb_param}')

    # Load datasets and create data loaders
    if train:
        trainset = TsagiSet(param, train=True)
        trainloader = DataLoader(trainset, batch_size=param['batch_size'],
                                 shuffle=True, pin_memory=True, num_workers=param['workers'])
    else:
        trainloader = None
    testset  = TsagiSet(param, train=False)
    testloader  = DataLoader(testset, batch_size=param['batch_size'],
                             shuffle=False, pin_memory=True, num_workers=param['workers'])

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
        optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'], weight_decay=param['l2_reg'])
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
