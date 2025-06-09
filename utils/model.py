from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), padding=1):
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # shortcut部分
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:   # ??????????????????
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        x = self.layer(x)
        x += self.shortcut(residual)
        return F.relu(x)


class ResNet18(nn.Module):
    def __init__(self, block, classes_num):
        super(ResNet18, self).__init__()
        self.in_channel = 64

        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64, [[1, 1], [1, 1]])
        self.layer2 = self._make_layer(block, 128, [[2, 1], [1, 1]])
        self.layer3 = self._make_layer(block, 256, [[2, 1], [1, 1]])
        self.layer4 = self._make_layer(block, 512, [[2, 1], [1, 1]])

        self.pool = nn.AvgPool2d((1, 1))
        self.fc = nn.Linear(512, classes_num)


    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channels, stride))
            self.in_channel = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)

        return out
