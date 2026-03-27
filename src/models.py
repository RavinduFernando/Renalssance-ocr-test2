"""
models.py

CRNN model definition for line-level OCR with CTC loss.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    A small CRNN:
    - CNN backbone to extract visual features
    - collapse height dimension
    - BiLSTM over width dimension
    - linear layer to character logits
    """

    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: number of characters including the CTC blank index.
        """
        super().__init__()

        # CNN: (N, 1, H, W) -> (N, C, H', W')
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # no pooling here; we keep width relatively large
        )

        # BiLSTM over time dimension (width)
        self.rnn = nn.LSTM(
            input_size=256,  # feature size after collapsing height
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
        )

        # Linear to logits
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: (N, 1, H, W)
        Returns:
            logits: (T, N, num_classes) for CTC
        """
        features = self.cnn(x)  # (N, C, H', W')
        n, c, h, w = features.size()

        # Collapse height by average pooling: (N, C, W)
        features = features.mean(dim=2)

        # Prepare for LSTM: (W, N, C)
        features = features.permute(2, 0, 1)

        seq, _ = self.rnn(features)  # (T=W, N, 2*hidden)
        logits = self.fc(seq)        # (T, N, num_classes)
        return logits

    def decode_greedy(self, logits, idx2char, blank_idx=0):
        """
        Greedy CTC decoding for a batch.
        Args:
            logits: (T, N, num_classes)
        Returns:
            List of decoded strings length N.
        """
        # Take argmax over classes
        probs = logits.log_softmax(dim=-1)
        max_indices = probs.argmax(dim=-1)  # (T, N)

        decoded = []
        max_indices = max_indices.transpose(0, 1)  # (N, T)
        for seq in max_indices:
            # collapse repeats and remove blanks
            prev = blank_idx
            chars = []
            for idx in seq.tolist():
                if idx != prev and idx != blank_idx:
                    chars.append(idx2char.get(idx, ""))
                prev = idx
            decoded.append("".join(chars))
        return decoded
