import torch.nn as nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    def __init__(self, drop_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(122880, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 2)
        self.drop = nn.Dropout(p=drop_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = nn.Softmax(dim=1)(x)
        return x


class VGG(nn.Module):
    def __init__(self, drop_size):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(30, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(7808, 500)
        self.fc2 = nn.Linear(500, 84)
        self.fc3 = nn.Linear(84, 2)
        self.drop = nn.Dropout(p=drop_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.drop(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = nn.Softmax(dim=1)(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(30, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.softmax(out)

        return out


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(SeparableConv2d, self).__init__()

        # groups=in_channels=out_channels
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                        padding=1, groups=in_channels, bias=bias)
        # 1*1 kernel size
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # print(x.dtype)
        x = x.to(torch.float32)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

# Residual Block
class residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(residual, self).__init__()

        self.sepConv1 = SeparableConv2d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.sepConv2 = SeparableConv2d(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # skip connection
        if out_channels != in_channels or stride != 1:
            # self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.sepConv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.sepConv2(y)
        y = self.bn2(y)
        if self.conv3:
            x = x.to(torch.float32)
            x = self.conv3(x)
        y = x + y
        y = self.relu2(y)
        return y

# BPR-STNet
class BPR(nn.Module):
    def __init__(self, num_classes=2):
        super(BPR, self).__init__()
        self.num_class = num_classes

        self.pw = nn.Conv2d(in_channels=30, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            residual(32, 32, 1),
            residual(32, 32, 1)
        )
        self.layer2 = nn.Sequential(
            residual(32, 64, 2),
            residual(64, 64, 1)
        )

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.pw(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        output = self.softmax(x)

        return output