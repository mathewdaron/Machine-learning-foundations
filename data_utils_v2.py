# =============================================================
# 文件：data_utils_v2.py
# 功能：CIFAR-10数据集加载、预处理、增强
# 输出：train_loader 和 test_loader，可直接传入训练
# =============================================================

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=128, data_dir='./data', num_workers=2):
    """
    加载CIFAR-10数据集，返回训练和测试DataLoader

    参数：
        batch_size：每批次样本数（默认128）
        data_dir：数据集存放目录（默认./data）
        num_workers：数据加载线程数（默认2）

    返回：
        train_loader：训练集DataLoader
        test_loader：测试集DataLoader

    CIFAR-10标准化参数（官方均值和标准差）：
        均值 mean = (0.4914, 0.4822, 0.4465)
        标准差 std  = (0.2023, 0.1994, 0.2010)
    """

    # ── 训练集变换：含数据增强 ──────────────────────────────────
    train_transform = transforms.Compose([
        # 随机裁剪：先padding4像素，再随机裁剪回32×32（增加位置多样性）
        transforms.RandomCrop(32, padding=4),
        # 随机水平翻转：50%概率，增加样本多样性
        transforms.RandomHorizontalFlip(),
        # 转为Tensor并归一化到[0,1]
        transforms.ToTensor(),
        # 标准化：减均值除标准差，使数据分布接近标准正态分布
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    # ── 测试集变换：不做数据增强，只做标准化 ────────────────────
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    # ── 加载数据集（首次运行会自动下载） ────────────────────────
    print("正在加载CIFAR-10数据集...")
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,       # 训练集：50000张
        download=True,    # 若本地无数据则自动下载
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,      # 测试集：10000张
        download=True,
        transform=test_transform
    )

    # ── 创建DataLoader（批量加载、打乱顺序）────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # 每轮训练打乱顺序
        num_workers=num_workers,
        pin_memory=True         # 锁页内存，加速GPU传输
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=True
    )

    # ── 打印数据集基本信息 ───────────────────────────────────────
    print(f"训练集大小: {len(train_dataset)} 张图片")
    print(f"测试集大小: {len(test_dataset)} 张图片")
    print(f"批次大小: {batch_size}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")

    return train_loader, test_loader


# ── 类别名称（CIFAR-10的10个类别）──────────────────────────────────
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# ── 直接运行测试 ─────────────────────────────────────────────────
if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders(batch_size=128)

    # 取一个batch查看形状
    images, labels = next(iter(train_loader))
    print(f"\n一个batch的图片形状: {images.shape}")  # [128, 3, 32, 32]
    print(f"一个batch的标签形状: {labels.shape}")    # [128]
    print(f"标签示例（前10个）: {labels[:10].tolist()}")
