import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import time
import os
import torch
from utils import load_yaml, oce, initialize
from vdp import loss_vdp


def train(param, device, trainloader, testloader, model, optimizer, epoch):
    loss_list = []
    for idx, (x, y) in enumerate(trainloader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        # Forward pass
        if param['vdp']:
            probs, var_prob = model(x, y)
            loss = loss_vdp(probs, var_prob, y, model, param)
        else:
            probs = model(x, y)
            loss = oce(probs, y, param)

        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), param['clip'])
        optimizer.step()

        loss_list.append(loss.item())

        if idx % int(len(trainloader)/4) == 0:
            if idx != len(trainloader)-1:
                total = idx*param['batch_size']
            else:
                total = (idx - 1)*param['batch_size'] + len(x)
            print(f'Epoch {epoch}: {total}/{len(trainloader.dataset)} {100*idx/len(trainloader):.0f}%, '
                  f'Loss: {np.mean(loss_list):.4f}')

    model.eval()
    with torch.no_grad():
        test_list = []
        tot_corr  = 0
        tot_num   = 0
        rmse      = 0
        for x, y in testloader:
            x, y  = x.to(device), y.to(device)
            if param['vdp']:
                pred, prob, var_prob = model.inference(x)
                loss                 = loss_vdp(prob, var_prob, y, model, param)
            else:
                pred, prob = model.inference(x)
                loss       = oce(prob, y, param)
            test_list.append(loss.item())
            y = y[:, 1:, :]
            rmse      += ((pred - y)**2).float().sum().item()
            tot_corr  += torch.eq(pred, y).float().sum().item()
            tot_num   += y.numel()
        acc   = 100*tot_corr/tot_num
        rmse  = math.sqrt(rmse/tot_num)
        mloss = np.mean(test_list)
        print(f'Epoch: {epoch}, Train Loss: {np.mean(loss_list):.6f},'
              f' Test Loss: {mloss:.6f}, Accuracy: {acc:.2f}%, RMSE: {rmse:.4f}')
    return mloss


def training(param, device, trainloader, testloader, model, optimizer):
    print('Start training')
    tac        = time.time()
    best_loss  = float('inf')
    best_epoch = 0
    for epoch in range(1, param['epochs'] + 1):
        tic       = time.time()
        test_loss = train(param, device, trainloader, testloader, model, optimizer, epoch)
        print(f'Epoch training time (s): {time.time() - tic}')
        if test_loss < best_loss:
            best_loss  = test_loss
            best_epoch = epoch
            checkpoint = model.state_dict()
            torch.save(checkpoint, f'models/{param["name"]}/weights.pt')
        elif test_loss > param['stop']*best_loss:
            print('Early stopping')
            print(f'Best epoch: {best_epoch}')
            print(f'Best loss: {best_loss:.6f}')
            break
    print(f'Training time (s): {time.time() - tac}')


def one_run(param):
    # Deterministic
    torch.manual_seed(param['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    # Declare CPU/GPU usage
    if param['gpu_number'] is not None:
        os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = param['gpu_number']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialization
    trainloader, testloader, model, optimizer = initialize(param, device, train=True)
    print(f'Nb of timestamps: {len(trainloader.dataset.times)}')
    print(f'Nb of sequences: {trainloader.dataset.total_seq}')
    print(f'Trainset length: {len(trainloader.dataset)}')
    print(f'Testset length: {len(testloader.dataset)}')
    print(f'Max nb of a/c: {trainloader.dataset.max_ac}')

    # Training
    training(param, device, trainloader, testloader, model, optimizer)


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    param = load_yaml('ed')
    one_run(param)


if __name__ == '__main__':
    main()
