from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvCell1D(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)) + x)


class DilatedCNN1D(nn.Module):
    """DCC-focused 1D CNN baseline over the full I/Q window."""

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 2,
        channels: int = 32,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.cells = nn.ModuleList(DilatedConvCell1D(channels, dilation) for dilation in dilations)
        feature_dim = 2 * channels * len(dilations)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        pooled = []
        for cell in self.cells:
            x = cell(x)
            avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
            max_values = F.adaptive_max_pool1d(x, 1).squeeze(-1)
            pooled.extend((avg, max_values))
        return self.classifier(torch.cat(pooled, dim=1))
