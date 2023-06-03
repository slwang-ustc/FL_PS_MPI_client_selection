import sys
sys.path.append('/data/slwang/FL_MPI_client_selection')
from config import cfg
import math
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride):
        super(BasicBlock, self).__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_planes, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes * BasicBlock.expansion, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes * BasicBlock.expansion, track_running_stats=False)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * BasicBlock.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_planes * BasicBlock.expansion, track_running_stats=False)
            )

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        out = residual + shortcut
        out = nn.ReLU(inplace=True)(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, num_block):
        super(ResNet, self).__init__()
        # torch.manual_seed(cfg['model_init_seed'])

        data_shape = cfg['data_shape']
        classes_size = cfg['classes_size']

        self.in_planes = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(data_shape[0], 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(block, 64, num_block[0], 1)
        self.layer2 = self._make_layer(block, 128, num_block[1], 2)
        self.layer3 = self._make_layer(block, 256, num_block[2], 2)
        self.layer4 = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, classes_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_planes, stride))
            self.in_planes = out_planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avg_pool(output)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        return output


def resnet18():
    """
    return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])
