import numpy as np
import pickle
import math
import time
import os
import torch
from utils import load_yaml, oce, initialize
from vdp import loss_vdp
from test import test, save_plot


def train(param, device, trainloader, testloader, model, optimizer, scheduler, epoch):
    loss_list, loss_val, nll_list, nll_val, kl_list = [], [], [], [], []
    class_nll_list = [[] for _ in range(param['nb_classes'])]
    for idx, (x, y) in enumerate(trainloader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        # Forward pass
        if param['vdp']:
            if param['no_zero']:
                y_       = y.clone()
                mask     = y_.ne(-1)*y_.ne(0)
                y_[mask] = torch.randint(1, param['nb_classes']+1, y_[mask].shape).to(device)
                probs, var_prob = model(x, y_)
            else:
                probs, var_prob = model(x, y)
            nll, kl, class_nll = loss_vdp(probs, var_prob, y, model, param, device)
            loss = nll + param['kl_factor']*kl
            nll_list.append(nll.item())
            kl_list.append(kl.item())
            for i in range(param['nb_classes']):
                class_nll_list[i].append(class_nll[i])
        else:
            probs = model(x, y)
            loss = oce(probs, y, param, device)

        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), param['clip'])
        optimizer.step()
        scheduler.step()

        loss_list.append(loss.item())

        if idx % max(int(len(trainloader)/4), 1) == 0:
            if idx != len(trainloader)-1:
                total = idx*param['batch_size']
            else:
                total = (idx - 1)*param['batch_size'] + len(x)
            if param['vdp']:
                print(f'Epoch {epoch}: {total}/{len(trainloader.dataset)} {100 * idx / len(trainloader):.0f}%, '
                      f'Loss: {np.mean(loss_list):.2f}, NLL: {np.mean(nll_list):.2f}, '
                      f'KL: {param["kl_factor"]*np.mean(kl_list):.2f}')
            else:
                print(f'Epoch {epoch}: {total}/{len(trainloader.dataset)} {100*idx/len(trainloader):.0f}%, '
                      f'Loss: {np.mean(loss_list):.2f}')

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
                nll, kl, class_nll   = loss_vdp(prob, var_prob, y, model, param, device)
                loss = nll + param['kl_factor']*kl
                nll_val.append(nll.item())
            else:
                pred, prob = model.inference(x)
                loss       = oce(prob, y, param, device)
            test_list.append(loss.item())
            loss_val.append(loss.item())
            if param['dataset'] == 'pirats':
                y     = y[:, 1:, :]
                mask  = y.ne(-1)
                if param['no_zero']:
                    mask = mask*y.ne(0)
                rmse += ((pred[mask] - y[mask])**2).float().sum().item()
                tot_corr += torch.eq(pred[mask], y[mask]).float().sum().item()
                tot_num  += mask.int().sum().item()
            else:
                tot_corr += torch.eq(pred, y).float().sum().item()
                tot_num += y.numel()
        acc   = 100*tot_corr/tot_num
        rmse  = math.sqrt(rmse/tot_num)
        mloss = np.mean(test_list)
        if param['dataset'] == 'pirats':
            if param['vdp']:
                print(f'Epoch: {epoch}, Train Loss: {np.mean(loss_list):.4f}, '
                      f'NLL: {np.mean(nll_list):.4f}, KL: {param["kl_factor"]*np.mean(kl_list):.4f}\n'
                      f'Test Loss: {mloss:.4f}, Accuracy: {acc:.2f}%, RMSE: {rmse:.4f}')
            else:
                print(f'Epoch: {epoch}, Train Loss: {np.mean(loss_list):.4f}, '
                      f'Test Loss: {mloss:.4f}, Accuracy: {acc:.2f}%, RMSE: {rmse:.4f}')
        elif param['dataset'] == 'mnist' or param['dataset'] == 'fashion':
            if param['vdp']:
                print(f'Epoch: {epoch}, Train Loss: {np.mean(loss_list):.4f}, '
                      f'NLL: {np.mean(nll_list):.4f}, KL: {param["kl_factor"]*np.mean(kl_list):.4f}\n'
                      f'Test Loss: {mloss:.4f}, Accuracy: {acc:.2f}%')
            else:
                print(f'Epoch: {epoch}, Train Loss: {np.mean(loss_list):.4f}, '
                      f'Test Loss: {mloss:.4f}, Accuracy: {acc:.2f}%')
    return mloss, loss_list, [np.mean(loss_val)], nll_list, [np.mean(nll_val)], kl_list, class_nll_list


def training(param, device, trainloader, testloader, model, optimizer, scheduler):
    print('Start training')
    tac        = time.time()
    best_loss  = float('inf')
    best_epoch = 0
    loss_full, loss_full_val, nll_full, nll_full_val, kl_full = [], [], [], [], []
    class_full = [[] for _ in range(param['nb_classes'])]
    for epoch in range(1, param['epochs'] + 1):
        tic       = time.time()
        res = train(param, device, trainloader, testloader, model, optimizer, scheduler, epoch)
        test_loss, loss_list, loss_val, nll_list, nll_val, kl_list, class_nll_list = res
        loss_full     = [*loss_full, *loss_list]
        loss_full_val = [*loss_full_val, *loss_val]
        nll_full      = [*nll_full, *nll_list]
        nll_full_val  = [*nll_full_val, *nll_val]
        kl_full       = [*kl_full, *kl_list]
        for i in range(len(class_full)):
            class_full[i] = [*class_full[i], *class_nll_list[i]]
        print(f'Epoch training time (s): {time.time() - tic}')
        if test_loss < best_loss:
            best_loss  = test_loss
            best_epoch = epoch
            print('Saving model')
            torch.save(model.state_dict(), f'models/{param["name"]}/weights.pt')
        elif (test_loss - best_loss)/abs(best_loss) > param['stop']:
            print('Early stopping')
            break
    print('Saving final model')
    torch.save(model.state_dict(), f'models/{param["name"]}/final.pt')
    print(f'Best epoch: {best_epoch}')
    print(f'Best loss: {best_loss:.6f}')
    print(f'Training time (s): {time.time() - tac}')
    return loss_full, loss_full_val, nll_full, nll_full_val, kl_full, class_full


def one_run(param):
    # Deterministic
    torch.manual_seed(param['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    # Declare CPU/GPU usage
    if param['gpu_number'] is not None:
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = param['gpu_number']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialization
    trainloader, testloader, model, optimizer, scheduler = initialize(param, device, train=True)
    if param['dataset'] == 'pirats':
        print(f'Nb of timestamps: {len(trainloader.dataset.times)}')
        print(f'Nb of sequences: {trainloader.dataset.total_seq}')
        print(f'Trainset length: {len(trainloader.dataset)}')
        print(f'Testset length: {len(testloader.dataset)}')
        print(f'Max nb of a/c: {trainloader.dataset.max_ac}')

    # Training
    res = training(param, device, trainloader, testloader, model, optimizer, scheduler)
    loss_full, loss_full_val, nll_full, nll_full_val, kl_full, class_full = res

    # Save
    with open( f'models/{param["name"]}/loss.pickle', 'wb') as f:
        print('Saving train loss list')
        pickle.dump(loss_full, f)
    with open( f'models/{param["name"]}/loss_val.pickle', 'wb') as f:
        print('Saving val loss list')
        pickle.dump(loss_full_val, f)
    with open( f'models/{param["name"]}/nll.pickle', 'wb') as f:
        print('Saving train nll list')
        pickle.dump(nll_full, f)
    with open( f'models/{param["name"]}/nll_val.pickle', 'wb') as f:
        print('Saving val nll list')
        pickle.dump(nll_full_val, f)
    with open( f'models/{param["name"]}/kl.pickle', 'wb') as f:
        print('Saving kl list')
        pickle.dump(kl_full, f)
    with open( f'models/{param["name"]}/class.pickle', 'wb') as f:
        print('Saving nll by class')
        pickle.dump(class_full, f)

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

    param = load_yaml('ed')
    one_run(param)


if __name__ == '__main__':
    main()
