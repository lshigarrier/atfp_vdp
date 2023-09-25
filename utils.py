import yaml
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import TsagiSet
from attention import TransformerED
from attention_vdp import TransformerED_VDP, ViT_VDP


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


def oce(probs, target, param, device):
    """
    Ordinal Cross-Entropy
    :param probs: batch_size x T_out x (nb_lon x nb_lat) x nb_classes
    :param target: batch_size x T_out x (nb_lon x nb_lat)
    :param param:
    :param device:
    :return: loss
    """
    probs  = probs.clamp(param['tol'], 1-param['tol'])
    target = target[:, 1:, :]
    nb_total, coef = 0, 0
    if param['balance']:
        nb_total = target.ne(-1).int().sum().item()
        coef     = torch.eq(target, 0)
    mask   = target.ne(-1).int()
    if param['no_zero']:
        mask = mask*target.ne(0).int()
    target = mask*target
    weight = torch.take(torch.tensor(param['weights']).to(device), target)
    target = F.one_hot(target, num_classes=param['nb_classes'])
    probs  = torch.matmul(target.unsqueeze(-2).float(), probs.unsqueeze(-1)).squeeze()
    loss   = -mask*weight*(1 - probs)**param['focus']*torch.log(probs)
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
            loss = loss*coef
    return loss.mean()


def load_partial_state_dict(model, other_state_dict):
    with torch.no_grad():
        for name, par in model.named_parameters():
            if name in other_state_dict:
                par.copy_(other_state_dict[name])


def initialize(param, device, train=True):
    # Print param
    for key in param:
        print(f'{key}: {param[key]}')
    print(f'device: {device}')
    # Create model
    print('Initialize model')
    if param['vdp']:
        if param['dataset'] == 'pirats':
            model = TransformerED_VDP(param, device)
        elif param['dataset'] == 'mnist' or param['dataset'] == 'fashion':
            model = ViT_VDP(param, device)
        else:
            raise NotImplementedError
    else:
        assert param['dataset'] == 'pirats'
        model = TransformerED(param, device)
    nb_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {nb_param}')

    # Load datasets and create data loaders
    transfos = transforms.Compose([
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor()
    ])
    trainset = None
    if param['dataset'] == 'pirats':
        trainset = TsagiSet(param, train=True)
    if train:
        if param['dataset'] == 'mnist':
            trainset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transfos)
        elif param['dataset'] == 'fashion':
            trainset = datasets.FashionMNIST('./data/fashion', train=True, download=True, transform=transfos)
        trainloader = DataLoader(trainset, batch_size=param['batch_size'],
                                 shuffle=True, pin_memory=True, num_workers=param['workers'])
    else:
        trainloader = [0]
    if param['dataset'] == 'pirats':
        testset = TsagiSet(param, train=False, quantiles=trainset.quantiles)
    elif param['dataset'] == 'mnist':
        testset = datasets.MNIST('./data/mnist', train=False, transform=transforms.ToTensor())
    elif param['dataset'] == 'fashion':
        testset = datasets.FashionMNIST('./data/fashion', train=False, transform=transforms.ToTensor())
    else:
        raise NotImplementedError
    testloader = DataLoader(testset, batch_size=param['batch_size'],
                            shuffle=False, pin_memory=True, num_workers=param['workers'])

    # Load model
    if param['load']:
        statedict = torch.load(f'models/{param["name"]}/{param["model"]}', map_location='cpu')
        model.load_state_dict(statedict)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    if param['pretrained']:
        statedict = torch.load(f'models/{param["name"]}/{param["pretrain"]}', map_location='cpu')
        load_partial_state_dict(model, statedict)
    model.to(device)

    # Set optimizer
    if param['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    elif param['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'], weight_decay=param['l2_reg'])
    else:
        raise NotImplementedError

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, param['Tmax']*len(trainloader))

    return trainloader, testloader, model, optimizer, scheduler


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
