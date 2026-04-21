"""数据加载工具：MNIST / CIFAR10 用 torchvision 自动下载；
IMDB 优先读 Kaggle CSV（data/IMDB Dataset.csv），找不到则 fallback Stanford 下载。
"""
import csv
import os
import random
import re
import tarfile
from collections import Counter

import requests
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_ROOT, exist_ok=True)


# -------------------- MNIST / CIFAR10 --------------------

def get_mnist_loaders(batch_size=128, num_workers=0):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_cifar10_loaders(batch_size=128, num_workers=0):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=train_tfm)
    test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=test_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


# -------------------- IMDB --------------------

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_DIR = os.path.join(DATA_ROOT, "aclImdb")
IMDB_CSV = os.path.join(DATA_ROOT, "IMDB Dataset.csv")  # Kaggle 下载的 CSV

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str):
    text = text.lower().replace("<br />", " ")
    return _TOKEN_RE.findall(text)


# ---------- 方式 A：Kaggle CSV ----------

def _load_imdb_from_csv():
    """从 Kaggle CSV 读取全部 50k 样本，随机 8:2 分 train/test。"""
    print(f"[IMDB] loading from Kaggle CSV: {IMDB_CSV}")
    samples = []
    with open(IMDB_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = 1 if row["sentiment"] == "positive" else 0
            samples.append((row["review"], label))
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    test_samples = samples[split:]
    return train_samples, test_samples


# ---------- 方式 B：Stanford tar.gz ----------

def _download_imdb():
    """若 aclImdb 不存在则下载并解压。"""
    if os.path.isdir(IMDB_DIR) and os.path.isdir(os.path.join(IMDB_DIR, "train")):
        return
    tar_path = os.path.join(DATA_ROOT, "aclImdb_v1.tar.gz")
    if not os.path.exists(tar_path):
        print(f"[IMDB] downloading {IMDB_URL} ...")
        with requests.get(IMDB_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r[IMDB] {done/1e6:.1f}/{total/1e6:.1f} MB", end="")
            print()
    print("[IMDB] extracting ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(DATA_ROOT)


def _load_imdb_split(split: str):
    """split: 'train' or 'test'，返回 [(text, label), ...]"""
    samples = []
    for label_name, label in [("pos", 1), ("neg", 0)]:
        folder = os.path.join(IMDB_DIR, split, label_name)
        for name in os.listdir(folder):
            if not name.endswith(".txt"):
                continue
            with open(os.path.join(folder, name), "r", encoding="utf-8") as f:
                samples.append((f.read(), label))
    return samples


def _build_vocab(train_tokens, max_size=20000, min_freq=2):
    counter = Counter()
    for toks in train_tokens:
        counter.update(toks)
    # 0: <pad>, 1: <unk>
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab


class IMDBDataset(Dataset):
    def __init__(self, samples, vocab, max_len=256):
        self.vocab = vocab
        self.max_len = max_len
        unk = vocab["<unk>"]
        self.data = []
        for text, label in samples:
            toks = _tokenize(text)[:max_len]
            ids = [vocab.get(t, unk) for t in toks]
            if len(ids) == 0:
                ids = [unk]
            self.data.append((ids, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, label = self.data[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def _collate_imdb(batch):
    # 按长度 pad 到 batch 内最长
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)  # pad=0
    labels = torch.zeros(len(batch), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)] = x
        labels[i] = y
    return ids, lengths, labels


def get_imdb_loaders(batch_size=64, max_len=256, num_workers=0):
    # 优先 Kaggle CSV，找不到再 fallback Stanford 下载
    if os.path.exists(IMDB_CSV):
        train_samples, test_samples = _load_imdb_from_csv()
    else:
        print("[IMDB] Kaggle CSV not found, fallback to Stanford download ...")
        _download_imdb()
        print("[IMDB] loading raw texts ...")
        train_samples = _load_imdb_split("train")
        test_samples = _load_imdb_split("test")
    print(f"[IMDB] train={len(train_samples)}  test={len(test_samples)}")

    print("[IMDB] building vocab ...")
    train_tokens = [_tokenize(t) for t, _ in train_samples]
    vocab = _build_vocab(train_tokens)
    print(f"[IMDB] vocab_size={len(vocab)}")

    train_set = IMDBDataset(train_samples, vocab, max_len=max_len)
    test_set = IMDBDataset(test_samples, vocab, max_len=max_len)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=_collate_imdb, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_imdb, num_workers=num_workers,
    )
    return train_loader, test_loader, vocab
