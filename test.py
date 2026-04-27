# =============================================================
# 文件：test.py
# 功能：单模型详细测试，输出规范化指标与可视化结果
# 用法：python test.py --model plain   (测试PlainCNN18)
#        python test.py --model resnet  (测试ResNet18)
# 输出：
#   ① 整体准确率 + 每类精度（终端）
#   ② 混淆矩阵图
#   ③ 预测样例展示图
#   ④ 【新增】每类识别准确率柱状图
#   ⑤ 【新增】预测置信度分布直方图
# =============================================================

import os
import sys
import argparse
import json

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')                # 无显示器环境使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils_v2 import get_data_loaders, CIFAR10_CLASSES
from models.plain_cnn_18layer import PlainCNN18
from models.resnet18_cifar10 import ResNet18

os.makedirs('results', exist_ok=True)


# ════════════════════════════════════════════════════════════════
# 核心：收集全部预测结果
# ════════════════════════════════════════════════════════════════
def get_all_predictions(model, loader, device):
    """
    遍历整个测试集，收集所有预测标签、真实标签和预测置信度
    返回：
        all_preds   : 所有预测标签，shape=(N,)
        all_labels  : 所有真实标签，shape=(N,)
        all_confs   : 每个样本预测类别的最高置信度，shape=(N,)
                      即 softmax 后取 max，范围 [0,1]
        sample_images: 第一个 batch 的原始图片（用于样例展示）
    """
    model.eval()
    all_preds    = []
    all_labels   = []
    all_confs    = []       # 每个样本的最高预测置信度
    sample_images = None

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # softmax 得到概率分布
            probs = torch.softmax(outputs, dim=1)         # [B, 10]
            # 取最高概率（即预测置信度）及对应类别
            conf, preds = probs.max(dim=1)                # [B], [B]

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_confs.append(conf.cpu().numpy())

            # 只保存第一个 batch 的图片用于样例可视化
            if batch_idx == 0:
                sample_images = images.cpu()

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_confs  = np.concatenate(all_confs)
    return all_preds, all_labels, all_confs, sample_images


# ════════════════════════════════════════════════════════════════
# 打印详细测试报告（终端输出）
# ════════════════════════════════════════════════════════════════
def print_report(all_preds, all_labels, model_name):
    """打印整体准确率和每类精度到终端，返回总准确率"""
    total_acc = np.mean(all_preds == all_labels) * 100

    print(f"\n{'='*60}")
    print(f"  模型：{model_name}")
    print(f"  测试集样本数：{len(all_labels)}")
    print(f"  整体 Top-1 Accuracy：{total_acc:.2f}%")
    print(f"{'='*60}")

    # sklearn 生成详细报告（每类 precision / recall / f1）
    report = classification_report(
        all_labels, all_preds,
        target_names=CIFAR10_CLASSES,
        digits=4
    )
    print("\n各类别详细指标（Precision / Recall / F1）：")
    print(report)

    # 每类准确率单独打印（更直观）
    print("每类别准确率：")
    for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
        mask    = all_labels == cls_idx
        cls_acc = np.mean(all_preds[mask] == all_labels[mask]) * 100
        correct = np.sum(all_preds[mask] == all_labels[mask])
        total   = mask.sum()
        print(f"  {cls_name:12s}: {cls_acc:6.2f}%  ({correct}/{total})")

    return total_acc


# ════════════════════════════════════════════════════════════════
# 原有图①：混淆矩阵
# ════════════════════════════════════════════════════════════════
def plot_confusion_matrix(all_preds, all_labels, model_name):
    """生成并保存混淆矩阵图"""
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CIFAR10_CLASSES
    )
    disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=45)
    ax.set_title(f'{model_name} — Confusion Matrix',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    save_path = f'results/confusion_matrix_{model_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存至 {save_path}")


# ════════════════════════════════════════════════════════════════
# 原有图②：预测样例展示
# ════════════════════════════════════════════════════════════════
def plot_sample_predictions(sample_images, all_preds, all_labels,
                             model_name, n=32):
    """
    展示前 n 张测试图片的预测结果
    标题：真实类别 / 预测类别，绿色=正确，红色=错误
    """
    # CIFAR-10 反标准化参数（还原图片原始色彩）
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])

    n    = min(n, len(sample_images))
    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.2))
    fig.suptitle(f'{model_name} — Sample Predictions\n'
                 f'(Green = Correct  |  Red = Wrong)',
                 fontsize=11, fontweight='bold')

    for i in range(n):
        ax = axes[i // cols][i % cols]

        # 反标准化：[C,H,W] → [H,W,C]，还原到 [0,1]
        img = sample_images[i].numpy().transpose(1, 2, 0)
        img = np.clip(img * std + mean, 0, 1)

        pred_label  = all_preds[i]
        true_label  = all_labels[i]
        is_correct  = (pred_label == true_label)
        title_color = 'green' if is_correct else 'red'

        ax.imshow(img)
        ax.set_title(
            f"T:{CIFAR10_CLASSES[true_label][:4]}\n"
            f"P:{CIFAR10_CLASSES[pred_label][:4]}",
            fontsize=7, color=title_color, fontweight='bold'
        )
        ax.axis('off')

    # 隐藏多余子图格
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis('off')

    plt.tight_layout()
    save_path = f'results/samples_{model_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 预测样例图已保存至 {save_path}")


# ════════════════════════════════════════════════════════════════
# 【新增图③】每类识别准确率柱状图
# ════════════════════════════════════════════════════════════════
def plot_per_class_accuracy(all_preds, all_labels, model_name):
    """
    横轴：10 个 CIFAR-10 类别
    纵轴：该类别的识别准确率（%）
    颜色：根据准确率高低区分（绿→黄→红渐变），一眼看出哪类容易出错
    辅助线：整体平均准确率参考虚线
    作用：直观展示模型在各类别上的强弱，体现分析深度
    """
    # 计算每类准确率
    class_accs = []
    for cls_idx in range(len(CIFAR10_CLASSES)):
        mask    = (all_labels == cls_idx)
        cls_acc = np.mean(all_preds[mask] == all_labels[mask]) * 100
        class_accs.append(cls_acc)

    overall_acc = np.mean(all_preds == all_labels) * 100  # 整体准确率

    # 根据准确率映射颜色：高准确率=绿，低准确率=红
    # 使用 RdYlGn 颜色映射（红→黄→绿）
    cmap     = plt.cm.RdYlGn
    # 将准确率归一化到 [0,1] 后映射颜色
    norm_acc = [(a - min(class_accs)) / (max(class_accs) - min(class_accs) + 1e-8)
                for a in class_accs]
    colors   = [cmap(v) for v in norm_acc]

    fig, ax = plt.subplots(figsize=(11, 6))

    # 按准确率排序，便于比较（这里保持原始类别顺序，更直观）
    x = np.arange(len(CIFAR10_CLASSES))
    bars = ax.bar(x, class_accs, width=0.6, color=colors,
                  edgecolor='white', linewidth=1.2, zorder=3)

    # 在每根柱子顶部标注数值
    for bar, acc in zip(bars, class_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{acc:.1f}%',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333333'
        )

    # 整体平均准确率虚线
    ax.axhline(
        y=overall_acc, color='#2C3E50', linewidth=1.8,
        linestyle='--', zorder=4,
        label=f'Overall Accuracy: {overall_acc:.2f}%'
    )

    # 坐标轴设置
    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES, fontsize=10, rotation=20, ha='right')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Class',        fontsize=12)
    ax.set_title(f'{model_name} — Per-Class Recognition Accuracy',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, axis='y', alpha=0.35, zorder=0)

    # 色彩图例（说明颜色含义）
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=min(class_accs),
                                                   vmax=max(class_accs)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.75)
    cbar.set_label('Accuracy (%)', fontsize=9)

    plt.tight_layout()
    save_path = f'results/per_class_acc_{model_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 每类准确率柱状图已保存至 {save_path}")


# ════════════════════════════════════════════════════════════════
# 【新增图④】预测置信度分布直方图
# ════════════════════════════════════════════════════════════════
def plot_confidence_distribution(all_confs, all_preds, all_labels, model_name):
    """
    画所有测试样本的预测置信度（softmax最大值）分布直方图
    分为：预测正确 / 预测错误 两组叠加对比
    横轴：置信度（0~1）
    纵轴：样本数量
    作用：
      - 正确预测集中在高置信度区间 → 模型预测有把握
      - 错误预测集中在低置信度区间 → 模型知道自己不确定
      - 若错误预测也有高置信度 → 模型过度自信（overconfident），训练不稳定
    """
    # 分离正确和错误样本的置信度
    correct_mask = (all_preds == all_labels)
    conf_correct = all_confs[correct_mask]    # 预测正确的置信度
    conf_wrong   = all_confs[~correct_mask]   # 预测错误的置信度

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{model_name} — Prediction Confidence Distribution',
                 fontsize=13, fontweight='bold')

    bins = np.linspace(0, 1, 26)   # 25个区间，每个区间宽度0.04

    # ── 左图：正确 vs 错误叠加对比直方图 ────────────────────
    axes[0].hist(conf_correct, bins=bins, alpha=0.75,
                 color='#2ECC71', label=f'Correct ({len(conf_correct)})',
                 edgecolor='white', linewidth=0.5)
    axes[0].hist(conf_wrong, bins=bins, alpha=0.75,
                 color='#E74C3C', label=f'Wrong ({len(conf_wrong)})',
                 edgecolor='white', linewidth=0.5)

    axes[0].set_xlabel('Confidence (Softmax Max Probability)', fontsize=10)
    axes[0].set_ylabel('Number of Samples', fontsize=10)
    axes[0].set_title('Correct vs Wrong Predictions', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)

    # 标注平均置信度
    axes[0].axvline(conf_correct.mean(), color='#27AE60', linewidth=1.5,
                    linestyle='--',
                    label=f'Correct mean: {conf_correct.mean():.3f}')
    if len(conf_wrong) > 0:
        axes[0].axvline(conf_wrong.mean(), color='#C0392B', linewidth=1.5,
                        linestyle='--',
                        label=f'Wrong mean: {conf_wrong.mean():.3f}')
    axes[0].legend(fontsize=8.5)

    # ── 右图：全体样本置信度 CDF（累积分布曲线）────────────
    # CDF 能直观看出"有多少比例的样本置信度高于某阈值"
    sorted_conf = np.sort(all_confs)
    cdf         = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)

    axes[1].plot(sorted_conf, cdf, color='#8E44AD', linewidth=2.2,
                 label='All samples CDF')
    axes[1].plot(np.sort(conf_correct),
                 np.arange(1, len(conf_correct) + 1) / len(conf_correct),
                 color='#2ECC71', linewidth=1.8, linestyle='--',
                 label='Correct CDF')
    if len(conf_wrong) > 0:
        axes[1].plot(np.sort(conf_wrong),
                     np.arange(1, len(conf_wrong) + 1) / len(conf_wrong),
                     color='#E74C3C', linewidth=1.8, linestyle='--',
                     label='Wrong CDF')

    # 参考线：置信度 0.5（随机猜测边界）
    axes[1].axvline(0.5, color='gray', linewidth=1.2, linestyle=':',
                    label='Conf = 0.5 (random)')
    axes[1].axvline(0.9, color='#F39C12', linewidth=1.2, linestyle=':',
                    label='Conf = 0.9 (high)')

    axes[1].set_xlabel('Confidence Threshold', fontsize=10)
    axes[1].set_ylabel('Cumulative Proportion', fontsize=10)
    axes[1].set_title('Confidence CDF\n(how confident the model is)', fontsize=11)
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    # ── 底部添加统计文字 ─────────────────────────────────────
    stats_text = (
        f"All samples: N={len(all_confs)}  |  "
        f"Mean conf: {all_confs.mean():.3f}  |  "
        f"Correct mean: {conf_correct.mean():.3f}  |  "
        f"Wrong mean: {conf_wrong.mean():.3f}  |  "
        f"High conf (>0.9): {(all_confs > 0.9).mean()*100:.1f}%"
    )
    fig.text(0.5, -0.01, stats_text, ha='center', va='top',
             fontsize=8.5, color='#555555',
             bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='#F0F0F0', edgecolor='#CCCCCC'))

    plt.tight_layout()
    save_path = f'results/confidence_dist_{model_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 置信度分布图已保存至 {save_path}")


# ════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='测试PlainCNN18或ResNet18')
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['plain', 'resnet'],
                        help="'plain'=PlainCNN18, 'resnet'=ResNet18")
    args = parser.parse_args()

    # ── 确定模型和权重路径 ────────────────────────────────────
    if args.model == 'plain':
        model_name  = 'plain_cnn'
        model       = PlainCNN18(num_classes=10)
        weight_path = 'results/best_plain_cnn.pth'
    else:
        model_name  = 'resnet18'
        model       = ResNet18(num_classes=10)
        weight_path = 'results/best_resnet18.pth'

    # ── 设备选择 ──────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n模型：{model_name}  |  设备：{device}")

    # ── 加载训练好的权重 ──────────────────────────────────────
    if not os.path.exists(weight_path):
        print(f"❌ 未找到权重文件：{weight_path}")
        print("   请先运行 train_v2.py 完成训练！")
        return

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    print(f"✅ 已加载权重：{weight_path}")

    # ── 加载测试集 ────────────────────────────────────────────
    _, test_loader = get_data_loaders(batch_size=128)

    # ── 推理（收集全部预测结果）──────────────────────────────
    print("\n正在推理测试集，请稍候...")
    all_preds, all_labels, all_confs, sample_images = \
        get_all_predictions(model, test_loader, device)

    # ── 输出终端报告 ──────────────────────────────────────────
    total_acc = print_report(all_preds, all_labels, model_name)

    # ── 生成所有图表 ──────────────────────────────────────────
    print("\n正在生成图表...")

    # 原有图①：混淆矩阵
    plot_confusion_matrix(all_preds, all_labels, model_name)

    # 原有图②：预测样例展示
    plot_sample_predictions(sample_images, all_preds, all_labels, model_name)

    # 新增图③：每类识别准确率柱状图
    plot_per_class_accuracy(all_preds, all_labels, model_name)

    # 新增图④：预测置信度分布直方图
    plot_confidence_distribution(all_confs, all_preds, all_labels, model_name)

    # ── 保存测试结果 JSON（供 compare.py 使用）───────────────
    result = {
        'model':       model_name,
        'total_acc':   total_acc,
        'per_class':   {},
        # 新增：置信度统计信息（供 compare.py 分析用）
        'conf_stats': {
            'mean_all':     float(all_confs.mean()),
            'mean_correct': float(all_confs[all_preds == all_labels].mean()),
            'mean_wrong':   float(all_confs[all_preds != all_labels].mean())
                           if (all_preds != all_labels).any() else 0.0,
            'high_conf_ratio': float((all_confs > 0.9).mean())
        }
    }
    for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
        mask    = (all_labels == cls_idx)
        cls_acc = float(np.mean(all_preds[mask] == all_labels[mask]) * 100)
        result['per_class'][cls_name] = cls_acc

    result_path = f'results/test_result_{model_name}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✅ 测试结果已保存至 {result_path}")

    # ── 最终汇总 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  测试完成！整体准确率：{total_acc:.2f}%")
    print(f"  所有图表已保存至 results/ 目录：")
    print(f"    confusion_matrix_{model_name}.png")
    print(f"    samples_{model_name}.png")
    print(f"    per_class_acc_{model_name}.png        ← 新增")
    print(f"    confidence_dist_{model_name}.png       ← 新增")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
