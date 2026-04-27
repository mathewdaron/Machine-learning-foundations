# =============================================================
# 文件：models/plain_cnn_18layer.py
# 功能：18层纯卷积CNN（无残差连接），用于与ResNet18对比
# 数据集：CIFAR-10（输入 3×32×32，输出10分类）
# 特点：深度高、无shortcut，容易出现梯度消失
# =============================================================

import torch
import torch.nn as nn

class PlainCNN18(nn.Module):
    """
    18层纯卷积CNN结构，严格按技术文档设计：
    - 全部使用 3×3 卷积，padding=1（保持空间尺寸）
    - 每层卷积后跟 BN（BatchNorm）+ ReLU
    - 通道数：64 → 128 → 256 → 512
    - 共4次MaxPool下采样，最后用自适应平均池化
    - 无任何残差/跳跃连接 → 这是与ResNet的核心区别
    """

    def __init__(self, num_classes=10):
        super(PlainCNN18, self).__init__()

        # ── 辅助函数：生成 Conv→BN→ReLU 组合层 ──────────────────
        def conv_bn_relu(in_ch, out_ch):
            """一个标准卷积块：3×3卷积 + 批归一化 + ReLU激活"""
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),   # 批归一化：稳定训练
                nn.ReLU(inplace=True)     # ReLU激活函数
            )

        # ── Stage 1：输入通道3→64，2层卷积 + MaxPool ─────────────
        # 输入: 3×32×32 → 输出: 64×16×16
        self.stage1 = nn.Sequential(
            conv_bn_relu(3, 64),          # 层1: 3→64
            conv_bn_relu(64, 64),         # 层2: 64→64
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32×32 → 16×16
        )

        # ── Stage 2：64→128，2层卷积 + MaxPool ───────────────────
        # 输入: 64×16×16 → 输出: 128×8×8
        self.stage2 = nn.Sequential(
            conv_bn_relu(64, 128),        # 层3: 64→128
            conv_bn_relu(128, 128),       # 层4: 128→128
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16×16 → 8×8
        )

        # ── Stage 3：128→256，2层卷积 + MaxPool ──────────────────
        # 输入: 128×8×8 → 输出: 256×4×4
        self.stage3 = nn.Sequential(
            conv_bn_relu(128, 256),       # 层5: 128→256
            conv_bn_relu(256, 256),       # 层6: 256→256
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8×8 → 4×4
        )

        # ── Stage 4：256→512，2层卷积 + MaxPool ──────────────────
        # 输入: 256×4×4 → 输出: 512×2×2
        self.stage4 = nn.Sequential(
            conv_bn_relu(256, 512),       # 层7: 256→512
            conv_bn_relu(512, 512),       # 层8: 512→512
            nn.MaxPool2d(kernel_size=2, stride=2)  # 4×4 → 2×2
        )

        # ── Stage 5：连续8层 512→512 卷积（层9~16）──────────────
        # 输入: 512×2×2 → 输出: 512×2×2（尺寸不变，padding=1保持）
        self.stage5 = nn.Sequential(
            conv_bn_relu(512, 512),       # 层9
            conv_bn_relu(512, 512),       # 层10
            conv_bn_relu(512, 512),       # 层11
            conv_bn_relu(512, 512),       # 层12
            conv_bn_relu(512, 512),       # 层13
            conv_bn_relu(512, 512),       # 层14
            conv_bn_relu(512, 512),       # 层15
            conv_bn_relu(512, 512),       # 层16
        )

        # ── 层17：自适应平均池化，将任意尺寸压缩为 1×1 ──────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出: 512×1×1

        # ── 层18：全连接层，512→10（10分类）─────────────────────
        self.classifier = nn.Linear(512, num_classes)

        # ── 权重初始化（Xavier初始化更稳定）──────────────────────
        self._initialize_weights()

    def _initialize_weights(self):
        """对所有卷积层和全连接层进行权重初始化"""
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
        """前向传播：依次通过各stage"""
        x = self.stage1(x)      # 3×32×32 → 64×16×16
        x = self.stage2(x)      # 64×16×16 → 128×8×8
        x = self.stage3(x)      # 128×8×8 → 256×4×4
        x = self.stage4(x)      # 256×4×4 → 512×2×2
        x = self.stage5(x)      # 512×2×2 → 512×2×2（深层）
        x = self.avgpool(x)     # 512×2×2 → 512×1×1
        x = torch.flatten(x, 1) # 512×1×1 → 512（展平）
        x = self.classifier(x)  # 512 → 10
        return x


# ── 简单测试（直接运行此文件可验证模型结构）──────────────────────
if __name__ == '__main__':
    model = PlainCNN18(num_classes=10)
    # 模拟一个batch：4张 3×32×32 的CIFAR-10图片
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    print("PlainCNN18 输出形状:", output.shape)  # 期望: torch.Size([4, 10])

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PlainCNN18 可训练参数量: {total_params:,}")
