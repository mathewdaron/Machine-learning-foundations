"""统一训练 / 测试 / 画图主程序。

用法示例:
    python train.py --model mlp
    python train.py --model cnn     --epochs 30
    python train.py --model resnet  --epochs 30
    python train.py --model gru     --epochs 8
    python train.py --model all

每个模型跑完会在 assets/ 下生成 4 张图:
    {model}_train_loss.png
    {model}_train_acc.png
    {model}_test_loss.png
    {model}_test_acc.png
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data_utils import get_mnist_loaders, get_cifar10_loaders, get_imdb_loaders
from models import MLP, CNN, ResNet, GRUClassifier

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


# -------------------- 训练 / 评估循环 --------------------

def _run_epoch(model, loader, criterion, optimizer, device, is_gru, train=True):
    model.train(train)
    total_loss, total_correct, total_num = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            if is_gru:
                x, lengths, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x, lengths)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(1) == y).sum().item()
            total_num += bs
    return total_loss / total_num, total_correct / total_num


def _save_curves(history, model_name):
    """history: dict with train_loss / train_acc / test_loss / test_acc (list)"""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    specs = [
        ("train_loss", "Train Loss", "loss"),
        ("train_acc", "Train Accuracy", "accuracy"),
        ("test_loss", "Test Loss", "loss"),
        ("test_acc", "Test Accuracy", "accuracy"),
    ]
    for key, title, ylabel in specs:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history[key], marker="o", linewidth=1.5)
        plt.title(f"{model_name.upper()} - {title}")
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out = os.path.join(ASSETS_DIR, f"{model_name}_{key}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  saved {out}")


def train_one(model_name, epochs, device):
    is_gru = False
    if model_name == "mlp":
        train_loader, test_loader = get_mnist_loaders(batch_size=128)
        model = MLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif model_name == "cnn":
        train_loader, test_loader = get_cifar10_loaders(batch_size=128)
        model = CNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    elif model_name == "resnet":
        train_loader, test_loader = get_cifar10_loaders(batch_size=128)
        model = ResNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    elif model_name == "gru":
        torch.backends.cudnn.enabled = False  # 避免 cuDNN RNN backward 报错
        train_loader, test_loader, vocab = get_imdb_loaders(batch_size=64, max_len=256)
        model = GRUClassifier(vocab_size=len(vocab)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        is_gru = True
    else:
        raise ValueError(model_name)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n==== training {model_name.upper()} | epochs={epochs} | device={device} ====")
    print(model)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device, is_gru, train=True)
        te_loss, te_acc = _run_epoch(model, test_loader, criterion, optimizer, device, is_gru, train=False)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        best_acc = max(best_acc, te_acc)
        print(f"[{model_name}] ep {ep:02d}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
              f"test loss {te_loss:.4f} acc {te_acc*100:.2f}% | "
              f"time {time.time()-t0:.1f}s")

    print(f"[{model_name}] best test acc = {best_acc*100:.2f}%")
    _save_curves(history, model_name)
    return best_acc


DEFAULT_EPOCHS = {"mlp": 10, "cnn": 30, "resnet": 30, "gru": 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    choices=["mlp", "cnn", "resnet", "gru", "all"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="不指定则按默认: mlp=10, cnn/resnet=30, gru=8")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    targets = ["mlp", "cnn", "resnet", "gru"] if args.model == "all" else [args.model]
    results = {}
    for name in targets:
        ep = args.epochs if args.epochs is not None else DEFAULT_EPOCHS[name]
        results[name] = train_one(name, ep, device)

    print("\n==== summary ====")
    for k, v in results.items():
        print(f"  {k:<6s} best test acc = {v*100:.2f}%")


if __name__ == "__main__":
    main()
