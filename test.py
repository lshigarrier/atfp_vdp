import os
import torch
import matplotlib.pyplot as plt
from utils import load_yaml, compute_probs, initialize
from plots import plot_pred


def test(param, device, testloader, model):
    with torch.no_grad():
        pred_tensor = torch.zeros(len(testloader.dataset), param['nb_lon'], param['nb_lat'], dtype=torch.int)
        true_tensor = torch.zeros(len(testloader.dataset), param['nb_lon'], param['nb_lat'], dtype=torch.int)

        t        = 0
        tot_corr = 0
        tot_num  = 0
        mse      = 0

        for idx, (x, label) in enumerate(testloader):
            x, label  = x.to(device), label.to(device)
            logits, b = model(x)
            probs     = compute_probs(logits, b, device)
            preds     = probs.argmax(dim=-1)
            pred_tensor[t:t+preds.shape[0], ...] = preds[:, -1, ...]
            true_tensor[t:t+label.shape[0], ...] = label[:, -1, ...]
            mse      += ((preds - label)**2).float().sum().item()
            tot_corr += torch.eq(preds, label).float().sum().item()
            tot_num  += label.numel()
            if idx % int(len(testloader)/4) == 0:
                print(f'Test: {idx*len(x)}/{len(testloader.dataset)} ({100.*idx/len(testloader):.0f}%)')
        acc = 100*tot_corr/tot_num
        mse = mse / tot_num
        print(f'Accuracy: {acc:.2f}%, MSE: {mse:.4f}')

        return pred_tensor, true_tensor


def one_test_run(param):
    # Deterministic
    torch.manual_seed(param['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    # Declare CPU/GPU usage
    if param['gpu_number'] is not None:
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = param['gpu_number']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
