import torch.nn as nn


def conv_bn_relu(in_ch, out_ch):
    """3x3 Conv + BN + ReLU，padding=1 保持特征图尺寸"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNN(nn.Module):
    """普通 6 层卷积 CNN，用于 CIFAR10 分类。

    3 个 block，每个 block 两层 3x3 卷积，通道 16->16->32->32->64->64，
    block 之间用 MaxPool 下采样 (32->16->8->4)。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # block1: 3 -> 16 -> 16
        self.block1 = nn.Sequential(
            conv_bn_relu(3, 16),
            conv_bn_relu(16, 16),
        )
        # block2: 16 -> 32 -> 32
        self.block2 = nn.Sequential(
            conv_bn_relu(16, 32),
            conv_bn_relu(32, 32),
        )
        # block3: 32 -> 64 -> 64
        self.block3 = nn.Sequential(
            conv_bn_relu(32, 64),
            conv_bn_relu(64, 64),
        )
        self.pool = nn.MaxPool2d(2, 2)  # 下采样只用 MaxPool

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.pool(self.block1(x))   # 32 -> 16
        x = self.pool(self.block2(x))   # 16 -> 8
        x = self.pool(self.block3(x))   # 8  -> 4
        return self.classifier(x)
