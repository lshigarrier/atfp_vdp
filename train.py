import numpy as np
import pickle
import math
import time
import os
import torch
from utils import load_yaml, oce, initialize
from vdp import loss_vdp
from test import test, save_plot


def train(param, device, trainloader, testloader, model, optimizer, epoch):
    loss_list = []
    nll_list  = []
    kl_list   = []
    for idx, (x, y) in enumerate(trainloader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        # Forward pass
        if param['vdp']:
            probs, var_prob = model(x, y)
            nll, kl = loss_vdp(probs, var_prob, y, model, param, device)
            loss = nll + param['kl_factor']*kl
            nll_list.append(nll.item())
            kl_list.append(kl.item())
        else:
            probs = model(x, y)
            loss = oce(probs, y, param, device)

        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), param['clip'])
        optimizer.step()

        loss_list.append(loss.item())

        if idx % max(int(len(trainloader)/4), 1) == 0:
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
                nll, kl              = loss_vdp(prob, var_prob, y, model, param, device)
                loss = nll + param['kl_factor']*kl
                # print(f'Statistics of predicted variances: {scipy.stats.describe(var_prob.flatten().detach().cpu())}')
            else:
                pred, prob = model.inference(x)
                loss       = oce(prob, y, param, device)
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
    return mloss, loss_list, nll_list, kl_list


def training(param, device, trainloader, testloader, model, optimizer):
    print('Start training')
    tac        = time.time()
    best_loss  = float('inf')
    best_epoch = 0
    loss_full, nll_full, kl_full = [], [], []
    for epoch in range(1, param['epochs'] + 1):
        tic       = time.time()
        test_loss, loss_list, nll_list, kl_list = train(param, device, trainloader, testloader, model, optimizer, epoch)
        loss_full = [*loss_full, *loss_list]
        nll_full = [*nll_full, *nll_list]
        kl_full = [*kl_full, *kl_list]
        print(f'Epoch training time (s): {time.time() - tic}')
        if test_loss < best_loss:
            best_loss  = test_loss
            best_epoch = epoch
            checkpoint = model.state_dict()
            torch.save(checkpoint, f'models/{param["name"]}/weights.pt')
        elif (test_loss - best_loss)/abs(best_loss) > param['stop']:
            print('Early stopping')
            break
    print(f'Best epoch: {best_epoch}')
    print(f'Best loss: {best_loss:.6f}')
    print(f'Training time (s): {time.time() - tac}')
    return loss_full, nll_full, kl_full


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
    loss_full, nll_full, kl_full = training(param, device, trainloader, testloader, model, optimizer)

    # Save
    with open( f'{param["fig_file"]}loss.pickle', 'wb') as f:
        pickle.dump(loss_full, f)
    with open( f'{param["fig_file"]}nll.pickle', 'wb') as f:
        pickle.dump(nll_full, f)
    with open( f'{param["fig_file"]}kl.pickle', 'wb') as f:
        pickle.dump(kl_full, f)

    # Test
    print('Start testing')
    varis = 0
    if param['vdp']:
        preds, truth, varis = test(param, device, testloader, model)
    else:
        preds, truth = test(param, device, testloader, model)

    save_plot(param, preds, truth, varis)


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    param = load_yaml('ed')
    one_run(param)


if __name__ == '__main__':
    main()
