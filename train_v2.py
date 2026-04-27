# =============================================================
# 文件：train_v2.py（新版，已整合梯度范数记录功能）
# 功能：统一训练框架，支持PlainCNN18与ResNet18切换训练
# 用法：python train_v2.py --model plain   (训练Plain CNN)
#        python train_v2.py --model resnet  (训练ResNet18)
# 输出：results/ 下的：
#   ① loss_{model}.png       — 训练/测试Loss曲线
#   ② acc_{model}.png        — 训练/测试Accuracy曲线
#   ③ lr_{model}.png         — 学习率变化曲线
#   ④ best_{model}.pth       — 最优模型权重
#   ⑤ log_{model}.json       — 完整训练日志
#   ⑥ grad_norm_{model}.json — 梯度范数日志（供compare.py使用）
# =============================================================

import os
import sys
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils_v2 import get_data_loaders
from models.plain_cnn_18layer import PlainCNN18
from models.resnet18_cifar10 import ResNet18

os.makedirs('results', exist_ok=True)


# ════════════════════════════════════════════════════════════════
# 训练配置（在这里修改超参数）
# ════════════════════════════════════════════════════════════════
CONFIG = {
    'batch_size':    128,

    # 【改动1/4】epoch：100 → 200
    # 原因：CosineAnnealingLR 需要足够长的周期才能发挥余弦退火的
    #       平滑衰减效果；同时给 ResNet18 更充分的收敛时间，
    #       避免上一次 100epoch 时 ResNet18 还未充分收敛就结束训练。
    'epochs':        200,

    # 优化器选择：'adam' 或 'sgd'
    'optimizer':     'sgd',
    # 初始学习率
    'lr':            0.1,
    # SGD动量（仅sgd有效）
    'momentum':      0.9,
    # 权重衰减（L2正则化，防止过拟合）
    'weight_decay':  5e-4,

    # 【改动2/4】移除 MultiStepLR 专用的 lr_milestones / lr_gamma
    # 原因：改用 CosineAnnealingLR 后这两个参数不再需要，
    #       保留在这里只会造成误导，故删除。
    # 删除前：'lr_milestones': [50, 75],
    # 删除前：'lr_gamma':      0.1,

    # 【改动2/4 续】新增 CosineAnnealingLR 专用参数
    # T_max  : 余弦退火的半周期 = 总 epoch 数
    #          LR 从 0.1 开始，在第 200 epoch 时降至 eta_min
    #          曲线是平滑的余弦曲线，没有突变台阶，对两个模型更公平
    # eta_min: LR 最低下限，防止学习率降到 0 导致完全停止学习
    'cosine_T_max':  200,
    'cosine_eta_min': 1e-4,

    # 数据目录
    'data_dir':      './data',
}


# ════════════════════════════════════════════════════════════════
# 单轮训练函数（同步收集每个batch的梯度范数，与上一版完全一致）
# ════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    训练一个epoch，返回平均Loss、准确率，以及本epoch的梯度范数列表

    梯度范数（Gradient Norm）说明：
      - 每次 loss.backward() 后，模型所有参数都有了梯度
      - 把全部参数梯度拼成一个大向量，计算它的L2范数
      - 范数越大 → 梯度越强，学习信号越充足
      - 范数持续接近0 → 梯度消失，模型停止学习
      - 对比两模型的梯度范数，可量化证明残差连接的效果

    返回：
        avg_loss         : 本epoch平均训练损失
        accuracy         : 本epoch训练准确率（%）
        batch_grad_norms : 本epoch每个batch的梯度L2范数列表
    """
    model.train()
    total_loss       = 0.0
    total_correct    = 0
    total_samples    = 0
    batch_grad_norms = []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # 计算本batch所有参数梯度的L2范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm   = p.grad.data.norm(2)
                total_norm  += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        batch_grad_norms.append(total_norm)

        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        _, predicted   = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy, batch_grad_norms


# ════════════════════════════════════════════════════════════════
# 单轮评估函数（与上一版完全一致）
# ════════════════════════════════════════════════════════════════
def evaluate(model, loader, criterion, device):
    """
    在测试集上评估模型，返回平均Loss和准确率
    评估时不计算梯度（with torch.no_grad()），节省内存和时间
    """
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss    += loss.item() * images.size(0)
            _, predicted   = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


# ════════════════════════════════════════════════════════════════
# 绘制并保存训练曲线
# ════════════════════════════════════════════════════════════════
def plot_and_save(history, model_name):
    """
    绘制Loss、Accuracy、LearningRate三条曲线并保存到results/

    【改动3/4】LR曲线：关闭对数坐标（yscale='log'）
    原因：MultiStepLR 是台阶式突变，log坐标能清晰显示台阶。
          CosineAnnealingLR 是平滑曲线，线性坐标能更直观展示
          余弦退火的优美弧线，换回线性坐标展示效果更好。
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # ── 图1：Loss曲线（与上一版完全一致）────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss',
             color='#2980B9', linewidth=2)
    plt.plot(epochs, history['test_loss'],  label='Test Loss',
             color='#E74C3C', linewidth=2, linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss',  fontsize=12)
    plt.title(f'{model_name} — Loss Curve', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/loss_{model_name}.png', dpi=150)
    plt.close()

    # ── 图2：Accuracy曲线（与上一版完全一致）────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_acc'], label='Train Acc',
             color='#2980B9', linewidth=2)
    plt.plot(epochs, history['test_acc'],  label='Test Acc',
             color='#E74C3C', linewidth=2, linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'{model_name} — Accuracy Curve', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/acc_{model_name}.png', dpi=150)
    plt.close()

    # ── 图3：Learning Rate曲线 ───────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['lr'], color='#27AE60', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title(f'{model_name} — Learning Rate Schedule (Cosine Annealing)',
              fontsize=13, fontweight='bold')

    # 【改动3/4】：注释掉 yscale('log')，改为线性坐标
    # plt.yscale('log')   # 旧版：对数坐标（适合台阶式MultiStepLR）
    # 新版：线性坐标，余弦曲线在线性轴上更美观、更直观
    plt.yscale('linear')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/lr_{model_name}.png', dpi=150)
    plt.close()

    print(f"✅ 训练曲线（Loss / Acc / LR）已保存至 results/")


# ════════════════════════════════════════════════════════════════
# 主训练流程
# ════════════════════════════════════════════════════════════════
def main():
    # ── 命令行参数解析 ────────────────────────────────────────
    parser = argparse.ArgumentParser(description='训练Plain CNN或ResNet18')
    parser.add_argument(
        '--model', type=str, default='resnet',
        choices=['plain', 'resnet'],
        help="选择模型：'plain'=PlainCNN18, 'resnet'=ResNet18"
    )
    args = parser.parse_args()

    # ── 确定模型名称和实例 ────────────────────────────────────
    if args.model == 'plain':
        model_name = 'plain_cnn'
        model      = PlainCNN18(num_classes=10)
    else:
        model_name = 'resnet18'
        model      = ResNet18(num_classes=10)

    # ── 设备选择（优先使用GPU）───────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 【改动4/4】启动信息：将 LR衰减行从打印 milestones 改为打印
    #            CosineAnnealingLR 的关键参数，方便训练时核对配置
    print(f"\n{'='*55}")
    print(f"  模型  : {model_name}")
    print(f"  设备  : {device}")
    print(f"  Epoch : {CONFIG['epochs']}")
    print(f"  优化器: {CONFIG['optimizer'].upper()}  "
          f"LR={CONFIG['lr']}  WD={CONFIG['weight_decay']}")
    # 旧版：print(f"  LR衰减: Epoch {CONFIG['lr_milestones']}  × {CONFIG['lr_gamma']}")
    # 新版：打印余弦退火参数
    print(f"  LR调度: CosineAnnealingLR  "
          f"T_max={CONFIG['cosine_T_max']}  "
          f"eta_min={CONFIG['cosine_eta_min']}")
    print(f"{'='*55}\n")

    model = model.to(device)

    # ── 数据加载 ──────────────────────────────────────────────
    train_loader, test_loader = get_data_loaders(
        batch_size=CONFIG['batch_size'],
        data_dir=CONFIG['data_dir']
    )

    # ── 损失函数 ──────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── 优化器（与上一版完全一致）────────────────────────────
    if CONFIG['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG['lr'],
            weight_decay=CONFIG['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG['lr'],
            momentum=CONFIG['momentum'],
            weight_decay=CONFIG['weight_decay']
        )

    # ── 【改动4/4 续】学习率调度器：MultiStepLR → CosineAnnealingLR ──
    #
    # 旧版（MultiStepLR）：
    #   scheduler = optim.lr_scheduler.MultiStepLR(
    #       optimizer,
    #       milestones=CONFIG['lr_milestones'],   # [50, 75]
    #       gamma=CONFIG['lr_gamma']              # 0.1
    #   )
    #   缺点：LR 在 milestone 处突然下降10倍，台阶式变化
    #         Plain CNN 刚好适应这种节奏，ResNet18 反而吃亏
    #
    # 新版（CosineAnnealingLR）：
    #   LR 按余弦函数从 lr_max 平滑衰减到 eta_min，无突变
    #   公式：LR(t) = eta_min + 0.5*(lr_max - eta_min)*(1 + cos(π*t/T_max))
    #   优势：
    #     ① 平滑无突变，对两个模型一视同仁，更公平
    #     ② 高LR阶段（前期）充分探索参数空间
    #     ③ 低LR阶段（后期）精细调整，让 ResNet18 有时间收敛
    #     ④ 是学术界训练 ResNet 的主流配置，写进报告更有说服力
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['cosine_T_max'],       # 半周期 = 总epoch数 200
        eta_min=CONFIG['cosine_eta_min']    # LR最低下限 1e-4
    )

    # ── 训练记录字典（与上一版完全一致）──────────────────────
    history = {
        'train_loss': [],
        'train_acc':  [],
        'test_loss':  [],
        'test_acc':   [],
        'lr':         [],
        # 每个epoch的梯度范数均值（代表梯度整体强度）
        'grad_mean':  [],
        # 每个epoch的梯度范数标准差（代表批次间梯度稳定性）
        'grad_std':   [],
    }

    best_acc  = 0.0
    best_path = f'results/best_{model_name}.pth'

    # ── 开始训练循环（与上一版完全一致）──────────────────────
    print("开始训练...\n")
    for epoch in range(1, CONFIG['epochs'] + 1):
        t0 = time.time()

        # 记录本轮开始时的学习率（scheduler.step()之前的值）
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc, batch_grads = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        # 更新学习率
        scheduler.step()

        # 记录各项指标
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        history['grad_mean'].append(float(np.mean(batch_grads)))
        history['grad_std'].append(float(np.std(batch_grads)))

        # 若本epoch测试准确率创新高，保存模型权重
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)

        # 打印本轮训练摘要
        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{CONFIG['epochs']}] "
            f"| LR: {current_lr:.6f} "
            f"| Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% "
            f"| Test  Loss: {test_loss:.4f}  Acc: {test_acc:.2f}% "
            f"| Best: {best_acc:.2f}% "
            f"| GradNorm: {history['grad_mean'][-1]:.4f} "
            f"| Time: {elapsed:.1f}s"
        )

    # ════════════════════════════════════════════════════════
    # 训练结束：保存所有日志和图表（与上一版完全一致）
    # ════════════════════════════════════════════════════════

    # 保存完整训练日志
    log_path = f'results/log_{model_name}.json'
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ 训练日志已保存至 {log_path}")

    # 保存梯度范数日志（compare.py专用）
    grad_log = {
        'epoch_mean': history['grad_mean'],
        'epoch_std':  history['grad_std'],
    }
    grad_path = f'results/grad_norm_{model_name}.json'
    with open(grad_path, 'w') as f:
        json.dump(grad_log, f, indent=2)
    print(f"✅ 梯度范数日志已保存至 {grad_path}")

    # 绘制训练曲线图
    plot_and_save(history, model_name)

    # 最终汇总
    print(f"\n{'='*55}")
    print(f"  训练完成！")
    print(f"  最优 Test Accuracy : {best_acc:.2f}%")
    print(f"  最优模型权重       : {best_path}")
    print(f"  训练日志           : {log_path}")
    print(f"  梯度范数日志       : {grad_path}")
    print(f"{'='*55}\n")


# ── 程序入口 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
