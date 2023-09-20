import os
import math
import pickle
import torch
import numpy as np
from vdp import loss_vdp
from utils import load_yaml, oce, initialize


def test(param, device, testloader, model):
    with torch.no_grad():

        if param['dataset'] == 'pirats':
            if param['predict_spot']:
                pred_tensor = torch.zeros(len(testloader.dataset), dtype=torch.int)
                true_tensor = torch.zeros(len(testloader.dataset), dtype=torch.int)
                if param['vdp']:
                    var_tensor = torch.zeros(len(testloader.dataset), dtype=torch.float)
            else:
                pred_tensor = torch.zeros(len(testloader.dataset), param['nb_lon'], param['nb_lat'], dtype=torch.int)
                true_tensor = torch.zeros(len(testloader.dataset), param['nb_lon'], param['nb_lat'], dtype=torch.int)
                if param['vdp']:
                    var_tensor = torch.zeros(len(testloader.dataset), param['nb_lon'], param['nb_lat'], dtype=torch.float)
        else:
            pred_tensor, true_tensor, var_tensor = 0, 0, 0

        test_list = []
        var_list  = [[], []]
        t         = 0
        tot_corr  = 0
        tot_num   = 0
        rmse      = 0

        for idx, (x, y) in enumerate(testloader):
            x, y  = x.to(device), y.to(device)
            if param['vdp']:
                pred, prob, var_prob = model.inference(x)
                indexes = pred.unsqueeze(-1).expand(*pred.shape, param['nb_classes']).long()
                var     = torch.take_along_dim(var_prob, indexes, dim=-1)[..., 0]
            else:
                pred, prob = model.inference(x)

            if param['dataset'] == 'pirats':
                if param['predict_spot']:
                    pred_tensor[t:t + pred.shape[0]] = pred[:, -1, 0].clone()
                    true_tensor[t:t + y.shape[0]]    = y[:, -1, 0].clone()
                    if param['vdp']:
                        var_tensor[t:t + pred.shape[0]] = var[:, -1, 0].clone()
                else:
                    pred_tensor[t:t+pred.shape[0], ...] = pred[:, -1, :].reshape(pred.shape[0],
                                                                                 param['nb_lon'],
                                                                                 param['nb_lat']).clone()
                    true_tensor[t:t+y.shape[0], ...]    = y[:, -1, :].reshape(y.shape[0],
                                                                              param['nb_lon'],
                                                                              param['nb_lat']).clone()
                    if param['vdp']:
                        if param['average']:
                            var_tensor[t:t + var.shape[0], ...] = var.mean(dim=1).reshape(var.shape[0],
                                                                                          param['nb_lon'],
                                                                                          param['nb_lat']).clone()
                        else:
                            var_tensor[t:t+var.shape[0], ...]  = var[:, -1, :].reshape(var.shape[0],
                                                                                       param['nb_lon'],
                                                                                       param['nb_lat']).clone()

            t += pred.shape[0]
            if param['vdp']:
                nll, kl = loss_vdp(prob, var_prob, y, model, param, device)
                loss = nll + param['kl_factor']*kl
            else:
                loss = oce(prob, y, param, device)
            test_list.append(loss.item())
            if param['dataset'] == 'pirats':
                y         = y[:, 1:, :]
                rmse     += ((pred - y)**2).float().sum().item()
            correct     = torch.eq(pred, y)
            tot_corr   += correct.float().sum().item()
            tot_num    += y.numel()
            if param['vdp']:
                var_list[0] = [*var_list, *var[correct].flatten().detach().tolist()]
                var_list[1] = [*var_list, *var[~correct].flatten().detach().tolist()]
            if idx % max(int(len(testloader)/4), 1) == 0:
                if idx != len(testloader)-1:
                    total = idx*param['batch_size']
                else:
                    total = (idx - 1)*param['batch_size'] + len(x)
                print(f'Test: {total}/{len(testloader.dataset)} ({100.*idx/len(testloader):.0f}%)')
        acc  = 100*tot_corr/tot_num
        rmse = math.sqrt(rmse/tot_num)
        if param['dataset'] == 'pirats':
            print(f'Test Loss: {np.mean(test_list):.6f}, Accuracy: {acc:.2f}%, RMSE: {rmse:.4f}')
        elif param['dataset'] == 'mnist' or param['dataset'] == 'fashion':
            print(f'Test Loss: {np.mean(test_list):.6f}, Accuracy: {acc:.2f}%')

        if param['vdp']:
            return pred_tensor, true_tensor, var_tensor, var_list
        else:
            return pred_tensor, true_tensor


def save_plot(param, preds, truth, varis, var_list):
    torch.save(preds, f'models/{param["name"]}/preds.pickle')
    torch.save(truth, f'models/{param["name"]}/truth.pickle')
    if param['vdp']:
        torch.save(varis, f'models/{param["name"]}/varis.pickle')
    var_correct = np.array(var_list[0])
    var_incorr  = np.array(var_list[1])
    torch.save(var_correct, f'models/{param["name"]}/var_corr.pickle')
    torch.save(var_incorr, f'models/{param["name"]}/var_incorr.pickle')


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
    trainloader, testloader, model, optimizer, scheduler = initialize(param, device, train=False)

    # Test
    print('Start testing')
    varis    = 0
    var_list = []
    if param['vdp']:
        preds, truth, varis, var_list = test(param, device, testloader, model)
    else:
        preds, truth = test(param, device, testloader, model)

    save_plot(param, preds, truth, varis, var_list)

def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    param = load_yaml('test_ed')
    one_test_run(param)


if __name__ == '__main__':
    main()
