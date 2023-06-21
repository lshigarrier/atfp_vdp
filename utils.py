import yaml
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TsagiSet
from attention import CongTrans


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


def oce(probs, target):
    """
    Ordinal Cross-Entropy
    :param target: batch_size x T_out x (nb_lon x nb_lat)
    :param probs: batch_size x T_out x (nb_lon x nb_lat) x nb_classes
    :return: loss
    """
    probs = probs.clamp(1e-6, 1-1e-6)
    idx   = target.unsqueeze(3).expand(*target.shape, probs.shape[3])
    probs = torch.take_along_dim(probs, idx, dim=3)[..., 0]
    # If the target is NOT 0 (i.e., congestion > 0), then the loss is multiplied by 100
    coef  = torch.logical_not(torch.eq(target, torch.zeros_like(target))).float()*1e2 + 1
    loss  = -torch.log(probs)*coef
    loss  = loss.mean()
    return loss


def initialize(param, device, train=True):
    # Create model
    model = CongTrans(param)

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
