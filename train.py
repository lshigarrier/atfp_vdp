import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import time
import os
import torch
from utils import load_yaml, compute_probs, oce, initialize


def train(device, trainloader, testloader, model, optimizer, epoch):
    loss_list = []
    for idx, (x, label) in enumerate(trainloader):
        optimizer.zero_grad()
        x, label = x.to(device), label.to(device)

        # Forward pass
        logits, b = model(x)

        # loss function
        loss = oce(logits, b, label, device)

        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        loss_list.append(loss.item())

        if idx % int(len(trainloader)/4) == 0:
            print(f'Epoch {epoch}: {idx*len(x)}/{len(trainloader.dataset)} {100*idx/len(trainloader):.0f}%, '
                  f'loss: {np.mean(loss_list):.4f}')

    model.eval()
    with torch.no_grad():
        tot_corr = 0
        tot_num  = 0
        mse      = 0
        for x, label in testloader:
            x, label  = x.to(device), label.to(device)
            logits, b = model(x)
            probs     = compute_probs(logits, b, device)
            preds     = probs.argmax(dim=-1)
            mse      += ((preds.flatten() - label.flatten())**2).float().sum().item()
            tot_corr += torch.eq(preds, label).float().sum().item()
            tot_num  += x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]
        acc = 100*tot_corr/tot_num
        mse = mse/tot_num
        print(f'Epoch: {epoch}, Loss: {np.mean(loss_list):.6f}, Accuracy: {acc:.2f}%, MSE: {mse:.4f}')


def training(param, device, trainloader, testloader, model, optimizer):
    print('Start training')
    tac = time.time()
    for epoch in range(1, param['epochs'] + 1):
        tic = time.time()
        train(device, trainloader, testloader, model, optimizer, epoch)
        print(f'Epoch training time (s): {time.time() - tic}')
    checkpoint = model.state_dict()
    torch.save(checkpoint, f'models/{param["name"]}/weights.pt')
    print(f'Training time (s): {time.time() - tac}')


def one_run(param):
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

    param = load_yaml('baseline')
    one_run(param)


if __name__ == '__main__':
    main()
