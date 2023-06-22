import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Initialize a residual block with two convolutions
    """
    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(hidden_size, out_size, kernel_size=(3, 3), padding=1)

    def convblock(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

    def forward(self, x): return x + self.convblock(x)  # skip connection


class CongCNN(nn.Module):
    def __init__(self, param, state_dim=6):
        super().__init__()
        channels     = param['channels']
        self.classes = param['nb_classes']
        self.conv1   = nn.Conv2d(state_dim, channels[0], kernel_size=(7, 7), stride=(1, 3), padding=(3, 3))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3), padding=(0, 1))
        self.conv2   = nn.Conv2d(channels[0], channels[1], kernel_size=(1, 9), stride=(1, 1), padding=0)
        self.blocks  = nn.ModuleList()
        self.blocks.append(ResBlock(channels[1], channels[1], channels[1]))
        self.conv3   = nn.Conv2d(channels[1], channels[2], kernel_size=(9, 9), stride=(1, 1), padding=0)
        self.blocks.append(ResBlock(channels[2], channels[2], channels[2]))
        self.conv4   = nn.Conv2d(channels[2], channels[3], kernel_size=(9, 9), stride=(1, 1), padding=0)
        self.blocks.append(ResBlock(channels[3], channels[3], channels[3]))
        self.conv5   = nn.Conv2d(channels[3], channels[4], kernel_size=(9, 9), stride=(1, 1), padding=0)
        self.blocks.append(ResBlock(channels[4], channels[4], channels[4]))
        self.conv6   = nn.Conv2d(channels[4], param['nb_classes']*param['T_out'],
                                 kernel_size=(7, 7), stride=(1, 1), padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.maxpool(x))
        x = F.relu(self.conv2(x))
        x = self.blocks[0](x)
        x = F.relu(self.conv3(x))
        x = self.blocks[1](x)
        x = F.relu(self.conv4(x))
        x = self.blocks[2](x)
        x = F.relu(self.conv5(x))
        x = self.blocks[3](x)
        x = self.conv6(x)
        # (batch size, T_out, nb_classes, nb_lon, nb_lat)
        x = x.view(x.shape[0], -1, self.classes, x.shape[2], x.shape[3])
        x[:, :, 2:, ...] = F.relu(x[:, :, 2:, ...])
        x[:, :, 1:, ...] = x[:, :, 1:, ...].cumsum(dim=2)
        return x