# =============================================================
# 文件：models/resnet18_cifar10.py
# 功能：适配CIFAR-10的标准ResNet18，含残差连接
# 数据集：CIFAR-10（输入 3×32×32，输出10分类）
# 特点：残差连接解决梯度消失，训练稳定，精度更高
# =============================================================

import torch
import torch.nn as nn

# ── BasicBlock：ResNet18的基础残差块 ─────────────────────────────
class BasicBlock(nn.Module):
    """
    ResNet18的基础构建单元（BasicBlock）
    结构：
        主路径：Conv→BN→ReLU→Conv→BN
        残差路径（shortcut）：直接连接 或 1×1卷积对齐维度
        输出：主路径 + shortcut → ReLU
    
    当 stride=2 或通道数改变时，shortcut需要用1×1卷积对齐维度
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        参数：
            in_channels：输入通道数
            out_channels：输出通道数
            stride：卷积步长（=2时做下采样）
        """
        super(BasicBlock, self).__init__()

        # ── 主路径：两个3×3卷积 ────────────────────────────────
        # 第一个卷积：可能改变通道数和空间尺寸（stride控制）
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二个卷积：保持通道数和尺寸不变（stride固定为1）
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ── 残差路径（shortcut）：当维度不匹配时用1×1卷积对齐 ──
        # 条件：stride≠1（空间尺寸改变）或 通道数改变
        self.shortcut = nn.Sequential()  # 默认：直接连接（恒等映射）
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 1×1卷积：对齐通道数和空间尺寸
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        前向传播：主路径 + shortcut → ReLU
        这就是残差连接的核心：out = F(x) + x
        """
        identity = x  # 保存输入，用于残差相加

        # 主路径计算
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差相加（这里是与PlainCNN的核心区别！）
        out = out + self.shortcut(identity)

        # 相加后再激活
        out = self.relu(out)
        return out


# ── ResNet18主体 ──────────────────────────────────────────────────
class ResNet18(nn.Module):
    """
    适配CIFAR-10的ResNet18
    结构：
        初始层：Conv(3→64)→BN→ReLU→MaxPool
        Stage1：2个BasicBlock(64→64)
        Stage2：2个BasicBlock(64→128, stride=2)
        Stage3：2个BasicBlock(128→256, stride=2)
        Stage4：2个BasicBlock(256→512, stride=2)
        全局平均池化 → 全连接(512→10)
    """

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        # ── 初始卷积层：处理原始图像 ──────────────────────────
        # 输入: 3×32×32 → 输出: 64×16×16
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # 注意：CIFAR-10图片小(32×32)，用kernel=3替代原始ResNet的7×7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32×32 → 16×16
        )

        # ── Stage1：64通道，不改变空间尺寸 ────────────────────
        # 输入: 64×16×16 → 输出: 64×16×16
        self.stage1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),   # 第1个残差块
            BasicBlock(64, 64, stride=1)    # 第2个残差块
        )

        # ── Stage2：64→128通道，空间尺寸减半 ──────────────────
        # 输入: 64×16×16 → 输出: 128×8×8
        self.stage2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),  # stride=2: 16×16 → 8×8
            BasicBlock(128, 128, stride=1)
        )

        # ── Stage3：128→256通道，空间尺寸减半 ─────────────────
        # 输入: 128×8×8 → 输出: 256×4×4
        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2), # stride=2: 8×8 → 4×4
            BasicBlock(256, 256, stride=1)
        )

        # ── Stage4：256→512通道，空间尺寸减半 ─────────────────
        # 输入: 256×4×4 → 输出: 512×2×2
        self.stage4 = nn.Sequential(
            BasicBlock(256, 512, stride=2), # stride=2: 4×4 → 2×2
            BasicBlock(512, 512, stride=1)
        )

        # ── 全局平均池化：将空间维度压为1×1 ──────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 512×2×2 → 512×1×1

        # ── 分类全连接层 ───────────────────────────────────────
        self.classifier = nn.Linear(512, num_classes)  # 512 → 10

        # ── 权重初始化 ─────────────────────────────────────────
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming初始化（适合ReLU激活函数）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播"""
        x = self.stem(x)     # 3×32×32 → 64×16×16
        x = self.stage1(x)   # 64×16×16 → 64×16×16
        x = self.stage2(x)   # 64×16×16 → 128×8×8
        x = self.stage3(x)   # 128×8×8 → 256×4×4
        x = self.stage4(x)   # 256×4×4 → 512×2×2
        x = self.avgpool(x)  # 512×2×2 → 512×1×1
        x = torch.flatten(x, 1)   # 512×1×1 → 512
        x = self.classifier(x)    # 512 → 10
        return x


# ── 简单测试 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    model = ResNet18(num_classes=10)
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    print("ResNet18 输出形状:", output.shape)  # 期望: torch.Size([4, 10])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet18 可训练参数量: {total_params:,}")
