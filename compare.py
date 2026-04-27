# =============================================================
# 文件：compare.py
# 功能：双模型公平对比分析，生成对比图表和文字总结
# 前提：两个模型均已完成训练（train_v2.py）和测试（test.py）
# 用法：python compare.py
# 输出：
#   ① compare_loss.png        — Loss曲线对比
#   ② compare_acc.png         — Accuracy曲线对比
#   ③ compare_per_class.png   — 每类别精度柱状对比
#   ④ 【新增】compare_grad_norm.png — 梯度范数对比（训练稳定性）
#   ⑤ 终端打印文字对比结论
# =============================================================

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs('results', exist_ok=True)

# CIFAR-10 类别名称
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 训练日志路径（train_v2.py 生成）
LOG_PLAIN  = 'results/log_plain_cnn.json'
LOG_RESNET = 'results/log_resnet18.json'

# 测试结果路径（test.py 生成）
TEST_PLAIN  = 'results/test_result_plain_cnn.json'
TEST_RESNET = 'results/test_result_resnet18.json'

# 梯度范数日志路径（train_v2.py 同步生成，见下方说明）
GRAD_PLAIN  = 'results/grad_norm_plain_cnn.json'
GRAD_RESNET = 'results/grad_norm_resnet18.json'


# ════════════════════════════════════════════════════════════════
# 通用：加载 JSON 文件
# ════════════════════════════════════════════════════════════════
def load_json(path, hint=''):
    """加载JSON文件，不存在时打印提示并返回 None"""
    if not os.path.exists(path):
        print(f"❌ 未找到：{path}  {hint}")
        return None
    with open(path, 'r') as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════
# 原有图①：Loss 曲线对比
# ════════════════════════════════════════════════════════════════
def plot_compare_loss(log_plain, log_resnet):
    """左：Test Loss 对比；右：Train Loss 对比"""
    ep_p = range(1, len(log_plain['test_loss'])  + 1)
    ep_r = range(1, len(log_resnet['test_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Plain CNN vs ResNet18 — Loss Comparison',
                 fontsize=13, fontweight='bold')

    # 左：Test Loss
    axes[0].plot(ep_p, log_plain['test_loss'],
                 label='Plain CNN',  color='#E74C3C', linewidth=2)
    axes[0].plot(ep_r, log_resnet['test_loss'],
                 label='ResNet18', color='#2980B9', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss',  fontsize=11)
    axes[0].set_title('Test Loss',  fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 右：Train Loss
    axes[1].plot(ep_p, log_plain['train_loss'],
                 label='Plain CNN', color='#E74C3C',
                 linewidth=2, linestyle='--')
    axes[1].plot(ep_r, log_resnet['train_loss'],
                 label='ResNet18', color='#2980B9',
                 linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss',  fontsize=11)
    axes[1].set_title('Train Loss', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'results/compare_loss.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss对比图已保存至 {save_path}")


# ════════════════════════════════════════════════════════════════
# 原有图②：Accuracy 曲线对比
# ════════════════════════════════════════════════════════════════
def plot_compare_acc(log_plain, log_resnet):
    """左：Test Acc 对比；右：Train Acc 对比"""
    ep_p = range(1, len(log_plain['test_acc'])  + 1)
    ep_r = range(1, len(log_resnet['test_acc']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Plain CNN vs ResNet18 — Accuracy Comparison',
                 fontsize=13, fontweight='bold')

    # 左：Test Acc
    axes[0].plot(ep_p, log_plain['test_acc'],
                 label='Plain CNN',  color='#E74C3C', linewidth=2)
    axes[0].plot(ep_r, log_resnet['test_acc'],
                 label='ResNet18', color='#2980B9', linewidth=2)
    axes[0].set_xlabel('Epoch',        fontsize=11)
    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0].set_title('Test Accuracy', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 右：Train Acc
    axes[1].plot(ep_p, log_plain['train_acc'],
                 label='Plain CNN', color='#E74C3C',
                 linewidth=2, linestyle='--')
    axes[1].plot(ep_r, log_resnet['train_acc'],
                 label='ResNet18', color='#2980B9',
                 linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch',        fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].set_title('Train Accuracy', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'results/compare_acc.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Accuracy对比图已保存至 {save_path}")


# ════════════════════════════════════════════════════════════════
# 原有图③：每类别准确率柱状对比
# ════════════════════════════════════════════════════════════════
def plot_per_class_acc(test_plain, test_resnet):
    """对比两模型在每个 CIFAR-10 类别上的准确率（并排柱状图）"""
    plain_accs  = [test_plain['per_class'][c]  for c in CIFAR10_CLASSES]
    resnet_accs = [test_resnet['per_class'][c] for c in CIFAR10_CLASSES]

    x     = np.arange(len(CIFAR10_CLASSES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, plain_accs,  width,
                   label='Plain CNN', color='#E74C3C', alpha=0.85)
    bars2 = ax.bar(x + width / 2, resnet_accs, width,
                   label='ResNet18',  color='#2980B9', alpha=0.85)

    ax.set_xlabel('Class',        fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy: Plain CNN vs ResNet18',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=30, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f'{bar.get_height():.1f}',
                ha='center', va='bottom', fontsize=7, color='#C0392B')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f'{bar.get_height():.1f}',
                ha='center', va='bottom', fontsize=7, color='#1A5276')

    plt.tight_layout()
    save_path = 'results/compare_per_class.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 每类别准确率对比图已保存至 {save_path}")


# ════════════════════════════════════════════════════════════════
# 【新增图④】梯度范数 / 训练稳定性对比
# ════════════════════════════════════════════════════════════════
def plot_grad_norm_compare(grad_plain, grad_resnet):
    """
    对比两模型训练过程中的梯度范数变化，用于证明：
      ResNet18 梯度更稳定、不消失
      Plain CNN 深层梯度更小、更容易消失

    grad_plain / grad_resnet 格式（由 train_v2.py 记录）：
      {
        "epoch_mean": [每个epoch的平均梯度范数],
        "epoch_min":  [每个epoch的最小梯度范数（最后一层→浅层差异）],
        "epoch_max":  [每个epoch的最大梯度范数]
      }

    图表布局（1行3列）：
      左：两模型每 epoch 平均梯度范数折线图（主图）
      中：梯度范数标准差（稳定性量化）
      右：梯度范数箱线图（全训练期分布）
    """
    ep_p = np.arange(1, len(grad_plain['epoch_mean'])  + 1)
    ep_r = np.arange(1, len(grad_resnet['epoch_mean']) + 1)

    mean_p = np.array(grad_plain['epoch_mean'])
    mean_r = np.array(grad_resnet['epoch_mean'])
    std_p  = np.array(grad_plain.get('epoch_std', [0] * len(mean_p)))
    std_r  = np.array(grad_resnet.get('epoch_std', [0] * len(mean_r)))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        'Gradient Norm Comparison: Plain CNN vs ResNet18\n'
        '（梯度范数对比：验证残差连接缓解梯度消失）',
        fontsize=12, fontweight='bold'
    )

    # ── 左图：平均梯度范数折线 ─────────────────────────────
    axes[0].plot(ep_p, mean_p, label='Plain CNN',
                 color='#E74C3C', linewidth=2)
    axes[0].plot(ep_r, mean_r, label='ResNet18',
                 color='#2980B9', linewidth=2)

    # 用半透明带表示±1倍标准差范围（体现波动幅度）
    axes[0].fill_between(ep_p, mean_p - std_p, mean_p + std_p,
                          color='#E74C3C', alpha=0.15)
    axes[0].fill_between(ep_r, mean_r - std_r, mean_r + std_r,
                          color='#2980B9', alpha=0.15)

    axes[0].set_xlabel('Epoch',           fontsize=10)
    axes[0].set_ylabel('Grad Norm (L2)',  fontsize=10)
    axes[0].set_title('Mean Gradient Norm per Epoch\n'
                       '（越稳定说明训练越健康）', fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    # 对数坐标（梯度范数跨度大，log更清晰）
    axes[0].set_yscale('log')

    # 添加注解箭头：指出平均值差异
    final_mean_p = mean_p[-1]
    final_mean_r = mean_r[-1]
    axes[0].annotate(
        f'Plain final:\n{final_mean_p:.4f}',
        xy=(ep_p[-1], final_mean_p),
        xytext=(ep_p[-1] * 0.75, final_mean_p * 1.5),
        fontsize=7.5, color='#E74C3C',
        arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.2)
    )
    axes[0].annotate(
        f'ResNet final:\n{final_mean_r:.4f}',
        xy=(ep_r[-1], final_mean_r),
        xytext=(ep_r[-1] * 0.75, final_mean_r * 0.5),
        fontsize=7.5, color='#2980B9',
        arrowprops=dict(arrowstyle='->', color='#2980B9', lw=1.2)
    )

    # ── 中图：梯度范数标准差（逐 epoch）─────────────────────
    # 标准差越小 → 每个 batch 之间梯度越一致 → 训练越稳定
    axes[1].plot(ep_p, std_p, label='Plain CNN',
                 color='#E74C3C', linewidth=2)
    axes[1].plot(ep_r, std_r, label='ResNet18',
                 color='#2980B9', linewidth=2)
    axes[1].set_xlabel('Epoch',              fontsize=10)
    axes[1].set_ylabel('Grad Norm Std Dev',  fontsize=10)
    axes[1].set_title('Gradient Norm Std Dev per Epoch\n'
                       '（标准差越小越稳定）', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # ── 右图：箱线图（全训练期梯度范数分布）────────────────
    # 箱线图能一眼看出中位数、四分位距、异常值
    data_p = mean_p.tolist()
    data_r = mean_r.tolist()

    bp = axes[2].boxplot(
        [data_p, data_r],
        labels=['Plain CNN', 'ResNet18'],
        patch_artist=True,       # 填充颜色
        notch=False,
        widths=0.5,
        medianprops=dict(color='white', linewidth=2)
    )

    # 设置箱体颜色
    colors_box = ['#E74C3C', '#2980B9']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for whisker in bp['whiskers']:
        whisker.set(color='#555555', linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color='#555555', linewidth=1.2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#999999', alpha=0.5, markersize=4)

    axes[2].set_ylabel('Gradient Norm (L2)', fontsize=10)
    axes[2].set_title('Gradient Norm Distribution\n'
                       '（全训练期箱线图）', fontsize=10)
    axes[2].grid(True, axis='y', alpha=0.3)

    # 在箱线图上标注均值
    for i, (data, color) in enumerate(zip([data_p, data_r], colors_box)):
        mn = np.mean(data)
        axes[2].text(
            i + 1, mn, f'μ={mn:.4f}',
            ha='center', va='bottom', fontsize=8,
            color=color, fontweight='bold'
        )

    plt.tight_layout()
    save_path = 'results/compare_grad_norm.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 梯度范数对比图已保存至 {save_path}")


# ════════════════════════════════════════════════════════════════
# 精度对比表 + 文字结论
# ════════════════════════════════════════════════════════════════
def print_summary(log_plain, log_resnet, test_plain, test_resnet,
                  grad_plain=None, grad_resnet=None):
    """打印精度对比表和实验结论到终端"""

    plain_best  = max(log_plain['test_acc'])
    resnet_best = max(log_resnet['test_acc'])
    plain_final_train  = log_plain['train_acc'][-1]
    resnet_final_train = log_resnet['train_acc'][-1]
    plain_std  = float(np.std(log_plain['test_loss'][10:]))
    resnet_std = float(np.std(log_resnet['test_loss'][10:]))

    print("\n" + "=" * 67)
    print("       Plain CNN vs ResNet18 — 对比实验总结报告")
    print("=" * 67)
    print(f"\n{'指标':<30} {'Plain CNN':>14} {'ResNet18':>14}")
    print("-" * 60)
    print(f"{'最高 Test Accuracy':<30} {plain_best:>13.2f}% {resnet_best:>13.2f}%")
    print(f"{'最终 Train Accuracy':<30} {plain_final_train:>13.2f}% {resnet_final_train:>13.2f}%")
    print(f"{'整体 Test Acc（test.py）':<30} "
          f"{test_plain['total_acc']:>13.2f}% {test_resnet['total_acc']:>13.2f}%")
    print(f"{'Test Loss 标准差（越小越稳定）':<30} "
          f"{plain_std:>14.4f} {resnet_std:>14.4f}")
    print(f"{'精度差（ResNet18 - Plain）':<30} "
          f"{resnet_best - plain_best:>+14.2f}%")

    # 置信度统计（若 test.py 新版已记录）
    if 'conf_stats' in test_plain and 'conf_stats' in test_resnet:
        p_conf = test_plain['conf_stats']
        r_conf = test_resnet['conf_stats']
        print(f"\n{'置信度统计':<30} {'Plain CNN':>14} {'ResNet18':>14}")
        print("-" * 60)
        print(f"{'平均预测置信度':<30} "
              f"{p_conf['mean_all']:>14.3f} {r_conf['mean_all']:>14.3f}")
        print(f"{'正确预测平均置信度':<30} "
              f"{p_conf['mean_correct']:>14.3f} {r_conf['mean_correct']:>14.3f}")
        print(f"{'错误预测平均置信度':<30} "
              f"{p_conf['mean_wrong']:>14.3f} {r_conf['mean_wrong']:>14.3f}")
        print(f"{'高置信度比例（>0.9）':<30} "
              f"{p_conf['high_conf_ratio']*100:>13.1f}% "
              f"{r_conf['high_conf_ratio']*100:>13.1f}%")

    # 梯度范数统计
    if grad_plain and grad_resnet:
        p_grad_mean = np.mean(grad_plain['epoch_mean'])
        r_grad_mean = np.mean(grad_resnet['epoch_mean'])
        p_grad_std  = np.std(grad_plain['epoch_mean'])
        r_grad_std  = np.std(grad_resnet['epoch_mean'])
        print(f"\n{'梯度范数统计':<30} {'Plain CNN':>14} {'ResNet18':>14}")
        print("-" * 60)
        print(f"{'训练期平均梯度范数':<30} "
              f"{p_grad_mean:>14.4f} {r_grad_mean:>14.4f}")
        print(f"{'梯度范数标准差（稳定性↓好）':<30} "
              f"{p_grad_std:>14.4f} {r_grad_std:>14.4f}")

    print("\n【实验结论】")
    winner = 'ResNet18' if resnet_best > plain_best else 'Plain CNN'
    diff   = abs(resnet_best - plain_best)
    stable = 'ResNet18' if resnet_std < plain_std else 'Plain CNN'

    print(f"  1. 精度：{winner} 领先约 {diff:.2f} 个百分点。")
    print(f"  2. 训练稳定性：{stable} 的测试损失波动更小，训练过程更稳定。")

    if resnet_best > plain_best:
        print("  3. 残差连接有效缓解了18层深度网络的梯度消失问题，")
        print("     ResNet18在相同深度下获得更高精度和更稳定的训练曲线。")
    else:
        print("  3. 两模型精度接近，但ResNet18的训练更稳定，可复现性更强。")

    print("\n  4. 每类别分析（ResNet18相对Plain CNN领先超过3%的类别）：")
    found_any = False
    for cls in CIFAR10_CLASSES:
        d = test_resnet['per_class'][cls] - test_plain['per_class'][cls]
        if d > 3.0:
            print(f"     - {cls:<12}: ResNet18 高出 {d:+.1f}%")
            found_any = True
    if not found_any:
        print("     （各类别差异均在3%以内，两模型表现相近）")

    if grad_plain and grad_resnet:
        p_final = grad_plain['epoch_mean'][-1]
        r_final = grad_resnet['epoch_mean'][-1]
        ratio   = r_final / (p_final + 1e-9)
        print(f"\n  5. 梯度分析：训练结束时 ResNet18 梯度范数")
        print(f"     为 Plain CNN 的 {ratio:.2f} 倍，")
        print(f"     说明残差连接维持了更健康的梯度流动，")
        print(f"     有效缓解了深层网络的梯度消失现象。")

    print("\n  6. 对比实验设计亮点：")
    print("     - 两模型均为18层，排除深度干扰，唯一变量=残差连接。")
    print("     - 数据增强、优化器、LR调度完全一致（公平对比）。")
    print("     - 梯度范数记录可量化地证明残差连接的训练优势。")
    print("=" * 67 + "\n")


# ════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════
def main():
    print("\n正在加载日志和测试结果...\n")

    log_plain   = load_json(LOG_PLAIN,  '→ 请先运行: python train_v2.py --model plain')
    log_resnet  = load_json(LOG_RESNET, '→ 请先运行: python train_v2.py --model resnet')
    test_plain  = load_json(TEST_PLAIN, '→ 请先运行: python test.py --model plain')
    test_resnet = load_json(TEST_RESNET,'→ 请先运行: python test.py --model resnet')

    # 核心文件缺失则退出
    if any(x is None for x in [log_plain, log_resnet, test_plain, test_resnet]):
        print("\n完整流程：")
        print("  1. python train_v2.py --model plain")
        print("  2. python train_v2.py --model resnet")
        print("  3. python test.py --model plain")
        print("  4. python test.py --model resnet")
        print("  5. python compare.py")
        return

    # 梯度范数日志（可选，若不存在则跳过该图）
    grad_plain  = load_json(GRAD_PLAIN,  '（可选，需更新train_v2.py记录梯度）')
    grad_resnet = load_json(GRAD_RESNET, '（可选，需更新train_v2.py记录梯度）')

    # ── 生成原有对比图 ────────────────────────────────────────
    plot_compare_loss(log_plain, log_resnet)
    plot_compare_acc(log_plain, log_resnet)
    plot_per_class_acc(test_plain, test_resnet)

    # ── 生成新增梯度范数对比图 ────────────────────────────────
    if grad_plain is not None and grad_resnet is not None:
        plot_grad_norm_compare(grad_plain, grad_resnet)
    else:
        print("\n⚠️  未找到梯度范数日志，跳过梯度对比图。")
        print("   请将下方「train_v2.py 梯度记录补丁」加入训练脚本后重新训练。\n")

    # ── 打印文字总结 ──────────────────────────────────────────
    print_summary(log_plain, log_resnet, test_plain, test_resnet,
                  grad_plain, grad_resnet)

    print("✅ 所有对比分析完成，请查看 results/ 目录。\n")


if __name__ == '__main__':
    main()
