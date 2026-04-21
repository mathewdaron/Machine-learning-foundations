import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """双向 2 层 GRU，用于 IMDB 二分类情感分析。

    词嵌入 128 维，隐藏层 128 维，双向拼接后 256 维，接全连接分类。
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, num_classes=2, pad_idx=0, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None):
        # x: (B, T)
        emb = self.embedding(x)  # (B, T, E)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h = self.gru(packed)
        else:
            _, h = self.gru(emb)
        # h: (num_layers*2, B, H)，取最后一层的正向+反向拼接
        h_fwd = h[-2]
        h_bwd = h[-1]
        feat = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2H)
        return self.fc(self.dropout(feat))
