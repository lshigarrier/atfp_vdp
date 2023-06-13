import os
import torch
from utils import load_yaml, compute_probs, initialize


def test(device, testset, model):
    with torch.no_grad():
        tot_corr = 0
        tot_num  = 0
        mse      = 0
        for idx, (x, label) in enumerate(testset):
            x, label  = x.to(device), label.to(device)
            logits, b = model(x)
            probs     = compute_probs(logits, b)
            preds     = probs.argmax(dim=-1)
            mse      += ((preds.flatten() - label.flatten())**2).float().sum().item()
            tot_corr += torch.eq(preds, label).float().sum().item()
            tot_num  += x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]
            if idx % int(len(testset)/4) == 0:
                print('Test: {}/{} ({:.0f}%)'.format(idx * len(x), len(testset.dataset), 100. * idx / len(testset)))
        acc = 100*tot_corr/tot_num
        mse = mse / tot_num
        print(f'Accuracy: {acc:.2f}%, MSE: {mse:.4f}')


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
    trainloader, testloader, model, optimizer = initialize(param, device)

    # Test
    test(device, testloader, model)


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    param = load_yaml('baseline')
    one_test_run(param)


if __name__ == '__main__':
    main()
