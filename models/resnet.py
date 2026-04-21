import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """标准 ResNet BasicBlock：两层 3x3 卷积 + 残差连接。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # 通道数变化时，shortcut 用 1x1 卷积对齐
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    """在 CNN 的基础上，把每个 block 的两层普通卷积换成一个 BasicBlock。
    通道: 3 -> 16 -> 32 -> 64，下采样用 MaxPool (32->16->8->4)。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = BasicBlock(3, 16)
        self.block2 = BasicBlock(16, 32)
        self.block3 = BasicBlock(32, 64)
        self.pool = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        return self.classifier(x)
