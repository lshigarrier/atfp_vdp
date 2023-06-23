import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_yaml, oce, initialize
from plots import plot_pred


def test(param, device, testloader, model):
    with torch.no_grad():
        pred_tensor = torch.zeros(len(testloader.dataset), param['nb_lon'], param['nb_lat'], dtype=torch.int)
        true_tensor = torch.zeros(len(testloader.dataset), param['nb_lon'], param['nb_lat'], dtype=torch.int)

        test_list = []
        t         = 0
        tot_corr  = 0
        tot_num   = 0
        rmse      = 0

        for idx, (x, y) in enumerate(testloader):
            x, y  = x.to(device), y.to(device)
            pred, prob = model.inference(x)
            pred_tensor[t:t+pred.shape[0], ...] = pred[:, -1, :].view(pred.shape[0],
                                                                      param['nb_lon'],
                                                                      param['nb_lat'])
            true_tensor[t:t+y.shape[0], ...] = y[:, -1, :].view(y.shape[0],
                                                                param['nb_lon'],
                                                                param['nb_lat'])
            t        += pred.shape[0]
            loss      = oce(prob, y, param)
            test_list.append(loss.item())
            y         = y[:, 1:, :]
            rmse     += ((pred - y)**2).float().sum().item()
            tot_corr += torch.eq(pred, y).float().sum().item()
            tot_num  += y.numel()
            if idx % int(len(testloader)/4) == 0:
                if idx != len(testloader)-1:
                    total = idx*param['batch_size']
                else:
                    total = (idx - 1)*param['batch_size'] + len(x)
                print(f'Test: {total}/{len(testloader.dataset)} ({100.*idx/len(testloader):.0f}%)')
        acc  = 100*tot_corr/tot_num
        rmse = math.sqrt(rmse/tot_num)
        print(f'Test Loss: {np.mean(test_list):.6f}, Accuracy: {acc:.2f}%, RMSE: {rmse:.4f}')

        return pred_tensor, true_tensor


def one_test_run(param):
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
    trainloader, testloader, model, optimizer = initialize(param, device, train=False)

    # Test
    print('Start testing')
    preds, truth = test(param, device, testloader, model)

    # Plot
    _ = plot_pred(preds, truth, param['nb_classes'])
    plt.show()


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    param = load_yaml('test')
    one_test_run(param)


if __name__ == '__main__':
    main()
