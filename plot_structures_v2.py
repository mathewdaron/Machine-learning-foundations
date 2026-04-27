# =============================================================
# 文件：plot_structures_v2.py
# 功能：绘制PlainCNN18和ResNet18的模型结构示意图
# 输出：results/structure_plain_cnn.png
#        results/structure_resnet18.png
# =============================================================

import os
import matplotlib
matplotlib.use('Agg')               # 无显示器环境下使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 确保输出目录存在
os.makedirs('results', exist_ok=True)


def draw_box(ax, x, y, width, height, text, color='#4A90D9',
             text_color='white', fontsize=8, radius=0.02):
    """
    在指定位置绘制一个圆角矩形块（代表一个网络层）
    参数：
        ax：matplotlib坐标轴
        x, y：矩形左下角坐标
        width, height：矩形尺寸
        text：方块内文字
        color：方块颜色
        text_color：文字颜色
        fontsize：字体大小
    """
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad={radius}",
        facecolor=color,
        edgecolor='white',
        linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha='center', va='center',
        color=text_color, fontsize=fontsize,
        fontweight='bold', wrap=True,
        multialignment='center'
    )


def draw_arrow(ax, x1, y1, x2, y2, color='#555555'):
    """在两个方块之间绘制箭头"""
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', color=color,
            lw=1.5, connectionstyle='arc3,rad=0'
        )
    )


# ════════════════════════════════════════════════════════════════
# 图1：PlainCNN18 结构图
# ════════════════════════════════════════════════════════════════
def plot_plain_cnn_structure():
    """绘制18层Plain CNN的结构示意图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    plt.title('18-Layer Plain CNN Architecture (CIFAR-10)',
              fontsize=14, fontweight='bold', pad=15, color='#222222')

    # ── 层定义：(x位置, 显示文字, 颜色) ───────────────────────
    bw = 1.1    # 方块宽度
    bh = 0.75   # 方块高度
    y  = 3.2    # 统一y坐标（水平排列）

    layers = [
        (0.2,  "Input\n3×32×32",             '#7F8C8D'),
        (1.4,  "Stage1\nConv×2\n64ch\n+Pool", '#2980B9'),
        (2.6,  "Stage2\nConv×2\n128ch\n+Pool",'#2980B9'),
        (3.8,  "Stage3\nConv×2\n256ch\n+Pool",'#2980B9'),
        (5.0,  "Stage4\nConv×2\n512ch\n+Pool",'#2980B9'),
        (6.2,  "Deep\nConv×2\n512ch",         '#1A6FA3'),
        (7.4,  "Deep\nConv×2\n512ch",         '#1A6FA3'),
        (8.6,  "Deep\nConv×2\n512ch",         '#1A6FA3'),
        (9.8,  "Deep\nConv×2\n512ch",         '#1A6FA3'),
        (11.0, "AvgPool\n512×1×1",            '#8E44AD'),
        (12.2, "FC\n512→10",                  '#27AE60'),
        (13.4, "Output\n10 cls",              '#E74C3C'),
    ]

    box_height = 1.2  # 加高方块以容纳多行文字
    for i, (x, text, color) in enumerate(layers):
        draw_box(ax, x, y - box_height/2, bw, box_height, text,
                 color=color, fontsize=7.5)
        if i < len(layers) - 1:
            draw_arrow(ax, x + bw, y, layers[i+1][0], y)

    # 图注说明
    ax.text(7, 1.0,
            "特点：无残差连接（No Shortcut）\n"
            "共18层卷积 → 深层梯度消失风险高\n"
            "卷积结构：Conv → BN → ReLU",
            ha='center', va='center', fontsize=9,
            color='#E74C3C', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FADBD8',
                      edgecolor='#E74C3C', alpha=0.8))

    # 层数标注
    ax.text(7, 5.8,
            "Layer 1-2      Layer 3-4     Layer 5-6     Layer 7-8     "
            "Layer 9-10  Layer 11-12  Layer 13-14  Layer 15-16    17        18",
            ha='center', va='center', fontsize=6.5, color='#555555')

    plt.tight_layout()
    save_path = 'results/structure_plain_cnn.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#F8F9FA')
    plt.close()
    print(f"✅ Plain CNN结构图已保存至 {save_path}")


# ════════════════════════════════════════════════════════════════
# 图2：ResNet18 结构图
# ════════════════════════════════════════════════════════════════
def plot_resnet18_structure():
    """绘制ResNet18的结构示意图（含残差连接示意）"""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    plt.title('ResNet18 Architecture (CIFAR-10) — with Residual Connections',
              fontsize=13, fontweight='bold', pad=15, color='#222222')

    bw   = 1.05   # 方块宽度
    bh   = 1.3    # 方块高度
    y    = 5.5    # 主路径y坐标

    # ── 主路径各模块 ────────────────────────────────────────────
    main_layers = [
        (0.2,  "Input\n3×32×32",            '#7F8C8D'),
        (1.35, "Stem\nConv3×3\n64ch\n+Pool",'#16A085'),
        (2.5,  "Stage1\nBlock×2\n64ch",     '#2980B9'),
        (3.65, "Stage2\nBlock×2\n128ch\n÷2",'#2980B9'),
        (4.8,  "Stage3\nBlock×2\n256ch\n÷2",'#2980B9'),
        (5.95, "Stage4\nBlock×2\n512ch\n÷2",'#2980B9'),
        (7.1,  "AvgPool\n512×1×1",          '#8E44AD'),
        (8.25, "FC\n512→10",                '#27AE60'),
        (9.4,  "Output\n10 cls",            '#E74C3C'),
    ]

    for i, (x, text, color) in enumerate(main_layers):
        draw_box(ax, x, y - bh/2, bw, bh, text,
                 color=color, fontsize=7.5)
        if i < len(main_layers) - 1:
            draw_arrow(ax, x + bw, y, main_layers[i+1][0], y)

    # ── BasicBlock放大示意图（右侧展示内部结构）─────────────────
    block_x = 11.0
    block_layers = [
        (block_x, 7.5, "Input x",         '#7F8C8D'),
        (block_x, 6.3, "Conv3×3\nBN+ReLU",'#2980B9'),
        (block_x, 5.0, "Conv3×3\nBN",     '#2980B9'),
        (block_x, 3.5, "Add (+)",          '#E67E22'),
        (block_x, 2.3, "ReLU\nOutput",    '#27AE60'),
    ]

    bw2, bh2 = 1.6, 0.75
    for (bx, by, bt, bc) in block_layers:
        draw_box(ax, bx, by - bh2/2, bw2, bh2, bt,
                 color=bc, fontsize=8)

    # 主路径箭头（BasicBlock内部）
    for i in range(len(block_layers) - 1):
        bx1, by1 = block_layers[i][0] + bw2/2,  block_layers[i][1]
        bx2, by2 = block_layers[i+1][0] + bw2/2, block_layers[i+1][1]
        draw_arrow(ax, bx1, by1 - bh2/2 - 0.02,
                   bx2, by2 + bh2/2 + 0.02)

    # shortcut箭头（绕过两个卷积层的弧线）
    ax.annotate(
        '', xy=(block_x + bw2 + 0.08, 3.5),
        xytext=(block_x + bw2 + 0.08, 7.5),
        arrowprops=dict(
            arrowstyle='->', color='#E74C3C', lw=2.0,
            connectionstyle='arc3,rad=-0.4'
        )
    )
    ax.text(block_x + bw2 + 0.85, 5.5,
            "shortcut\n(残差连接)",
            ha='center', va='center', fontsize=8,
            color='#E74C3C', fontweight='bold')

    # BasicBlock标题
    ax.text(block_x + bw2/2, 8.5,
            "BasicBlock 内部结构",
            ha='center', va='center', fontsize=9,
            color='#222222', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#EBF5FB', edgecolor='#2980B9'))

    # 虚线框圈出BasicBlock区域
    import matplotlib.patches as mpatches
    rect = mpatches.FancyBboxPatch(
        (block_x - 0.15, 1.8), bw2 + 1.3, 7.0,
        boxstyle="round,pad=0.1",
        facecolor='none', edgecolor='#2980B9',
        linewidth=1.5, linestyle='--'
    )
    ax.add_patch(rect)

    # ── 图注说明 ─────────────────────────────────────────────────
    ax.text(5.0, 0.8,
            "特点：含残差连接（Shortcut）\n"
            "当维度不匹配时，shortcut使用1×1卷积对齐\n"
            "梯度可直接通过shortcut反向传播 → 有效缓解梯度消失",
            ha='center', va='center', fontsize=9,
            color='#1A5276', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#D6EAF8',
                      edgecolor='#2980B9', alpha=0.85))

    plt.tight_layout()
    save_path = 'results/structure_resnet18.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#F8F9FA')
    plt.close()
    print(f"✅ ResNet18结构图已保存至 {save_path}")


# ── 主程序 ────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("正在绘制模型结构图...")
    plot_plain_cnn_structure()
    plot_resnet18_structure()
    print("\n所有结构图已生成完毕，请查看 results/ 目录。")
