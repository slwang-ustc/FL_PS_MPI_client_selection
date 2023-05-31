import torch
import torch.nn as nn
from config import cfg


def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


class CNN(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size):
        super(CNN, self).__init__()

        torch.manual_seed(cfg['model_init_seed'])

        # head
        blocks = [nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1)]

        for i in range(len(hidden_size) - 1):
            blocks.extend([
                nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])

        # classifier
        blocks.extend([
            nn.Flatten(),
            nn.Linear(
                int(hidden_size[-1] * cfg['data_shape'][1] * cfg['data_shape'][2] / (4 ** (len(hidden_size)-1))), 
                classes_size
            )
        ])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        output = self.blocks(x)
        return output
