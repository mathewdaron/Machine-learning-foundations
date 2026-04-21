"""用 matplotlib 绘制 4 个模型的结构示意图，保存到 assets/*_structure.png。
不依赖 graphviz/torchviz，纯画方框箭头。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


# -------------------- 基础绘制函数 --------------------

COLOR_INPUT = "#BFD7EA"
COLOR_CONV = "#F7C59F"
COLOR_POOL = "#EFEFD0"
COLOR_FC = "#FF9F68"
COLOR_RES = "#B5EAD7"
COLOR_EMB = "#C7CEEA"
COLOR_RNN = "#FFDAC1"
COLOR_OUT = "#E56B6F"


def _box(ax, x, y, w, h, text, color, fontsize=9):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2, edgecolor="#333", facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, family="monospace")


def _arrow(ax, x1, y1, x2, y2, color="#555"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=14,
        linewidth=1.2, color=color,
    ))


def _vertical_chain(ax, blocks, x=0.5, top=0.96, w=3.0, h=0.35, gap=0.12):
    """blocks: list of (text, color)。从上到下画，并用箭头连接。"""
    y = top
    centers = []
    for text, color in blocks:
        _box(ax, x - w / 2, y - h, w, h, text, color)
        centers.append((x, y - h / 2, y - h, y))  # (cx, cy, bottom, top)
        y = y - h - gap
    for i in range(len(centers) - 1):
        cx, _, bottom, _ = centers[i]
        _, _, _, ntop = centers[i + 1]
        _arrow(ax, cx, bottom, cx, ntop)
    return centers


def _setup_ax(figsize=(5, 9), title=""):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-2, 2)
    # ylim 由调用方决定
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    return fig, ax


# -------------------- 四个模型 --------------------

def plot_mlp():
    fig, ax = _setup_ax(figsize=(5, 8), title="MLP (MNIST)")
    blocks = [
        ("Input  (B, 1, 28, 28)", COLOR_INPUT),
        ("Flatten -> 784", COLOR_POOL),
        ("Linear(784, 256) + ReLU + Dropout", COLOR_FC),
        ("Linear(256, 128) + ReLU + Dropout", COLOR_FC),
        ("Linear(128, 10)", COLOR_OUT),
        ("Output logits (B, 10)", COLOR_INPUT),
    ]
    _vertical_chain(ax, blocks, x=0, top=3.4, w=3.6, h=0.45, gap=0.15)
    ax.set_ylim(-1.0, 3.8)
    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "mlp_structure.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def plot_cnn():
    fig, ax = _setup_ax(figsize=(6, 10), title="CNN (CIFAR10)  3 blocks x 2 conv")
    blocks = [
        ("Input  (B, 3, 32, 32)", COLOR_INPUT),
        ("Block1:  Conv3x3(3->16) + BN + ReLU\n         Conv3x3(16->16) + BN + ReLU", COLOR_CONV),
        ("MaxPool 2x2   (16, 16, 16)", COLOR_POOL),
        ("Block2:  Conv3x3(16->32) + BN + ReLU\n         Conv3x3(32->32) + BN + ReLU", COLOR_CONV),
        ("MaxPool 2x2   (32, 8, 8)", COLOR_POOL),
        ("Block3:  Conv3x3(32->64) + BN + ReLU\n         Conv3x3(64->64) + BN + ReLU", COLOR_CONV),
        ("MaxPool 2x2   (64, 4, 4)", COLOR_POOL),
        ("AdaptiveAvgPool -> (64,)", COLOR_POOL),
        ("Linear(64, 10)", COLOR_OUT),
        ("Output logits (B, 10)", COLOR_INPUT),
    ]
    _vertical_chain(ax, blocks, x=0, top=5.8, w=4.6, h=0.55, gap=0.15)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.2, 6.2)
    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "cnn_structure.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def plot_resnet():
    fig, ax = _setup_ax(figsize=(7, 10), title="ResNet (CIFAR10)  3 BasicBlocks")
    # 主干
    blocks = [
        ("Input  (B, 3, 32, 32)", COLOR_INPUT),
        ("BasicBlock1 (3 -> 16)", COLOR_RES),
        ("MaxPool 2x2   (16, 16, 16)", COLOR_POOL),
        ("BasicBlock2 (16 -> 32)", COLOR_RES),
        ("MaxPool 2x2   (32, 8, 8)", COLOR_POOL),
        ("BasicBlock3 (32 -> 64)", COLOR_RES),
        ("MaxPool 2x2   (64, 4, 4)", COLOR_POOL),
        ("AdaptiveAvgPool -> (64,)", COLOR_POOL),
        ("Linear(64, 10)", COLOR_OUT),
        ("Output logits (B, 10)", COLOR_INPUT),
    ]
    _vertical_chain(ax, blocks, x=-1.2, top=5.8, w=3.6, h=0.55, gap=0.15)

    # 右侧画 BasicBlock 展开图
    bx = 1.8
    bw = 2.4
    bh = 0.45
    bgap = 0.12
    sub = [
        ("BasicBlock (in -> out)", COLOR_RES),
        ("Conv3x3 + BN + ReLU", COLOR_CONV),
        ("Conv3x3 + BN", COLOR_CONV),
        ("Add  (identity / 1x1 shortcut)", COLOR_RES),
        ("ReLU", COLOR_FC),
    ]
    y = 3.2
    centers = []
    for text, color in sub:
        _box(ax, bx - bw / 2, y - bh, bw, bh, text, color, fontsize=8)
        centers.append((bx, y - bh, y))
        y = y - bh - bgap
    for i in range(len(centers) - 1):
        cx, bottom, _ = centers[i]
        _, _, ntop = centers[i + 1]
        _arrow(ax, cx, bottom, cx, ntop)
    # 残差连接曲线：从第 1 个 conv 之前绕到 add 框
    ax.annotate("",
                xy=(bx + bw / 2 + 0.05, centers[3][1] + bh / 2),
                xytext=(bx + bw / 2 + 0.05, centers[0][1] - 0.02),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.6",
                                color="#c0392b", linewidth=1.3))
    ax.text(bx + bw / 2 + 0.35, (centers[0][1] + centers[3][1]) / 2,
            "shortcut", color="#c0392b", fontsize=8, rotation=90, va="center")

    ax.set_xlim(-3.5, 3.8)
    ax.set_ylim(-1.2, 6.2)
    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "resnet_structure.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def plot_gru():
    fig, ax = _setup_ax(figsize=(6, 9), title="Bi-GRU (IMDB)  2 layers, hidden=128")
    blocks = [
        ("Input token ids  (B, T)", COLOR_INPUT),
        ("Embedding(vocab, 128, pad_idx=0)", COLOR_EMB),
        ("GRU layer 1   bi-directional, hidden=128", COLOR_RNN),
        ("GRU layer 2   bi-directional, hidden=128", COLOR_RNN),
        ("Concat last h_fwd & h_bwd  -> (B, 256)", COLOR_POOL),
        ("Dropout", COLOR_POOL),
        ("Linear(256, 2)", COLOR_OUT),
        ("Output logits (B, 2)", COLOR_INPUT),
    ]
    _vertical_chain(ax, blocks, x=0, top=4.6, w=4.4, h=0.5, gap=0.15)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.2, 5.0)
    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "gru_structure.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    plot_mlp()
    plot_cnn()
    plot_resnet()
    plot_gru()


if __name__ == "__main__":
    main()
