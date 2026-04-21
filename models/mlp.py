import torch.nn as nn


class MLP(nn.Module):
    """两个隐藏层的简单 MLP，用于 MNIST 手写数字分类。

    输入:  (B, 1, 28, 28)
    输出:  (B, 10)
    """

    def __init__(self, in_dim=28 * 28, hidden1=256, hidden2=128, num_classes=10, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)
