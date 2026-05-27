from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResBlock1D


class _ScaleEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1D(channels, channels),
            ResBlock1D(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool1d(self.blocks(self.stem(x)), 1).squeeze(-1)


class MultiScalePyramidNet(nn.Module):
    """Multi-scale feature pyramid for signals with variable symbol rate.

    Processes the signal in parallel at five explicit temporal scales (AvgPool
    stride 1x, 2x, 4x, 8x, 16x). At scale k the downsampled branch operates at
    approximately one sample per symbol when the true OSR is k, giving each branch
    a different effective symbol-rate hypothesis. Independent ScaleEncoders produce
    fixed-size feature vectors that are concatenated and classified.

    The parallel multi-scale pyramid structure is inspired by feature pyramid
    networks for multi-resolution representation.

    Citation:
        Lin, Tsung-Yi, et al. "Feature Pyramid Networks for Object Detection."
        *Proceedings of the IEEE Conference on Computer Vision and Pattern
        Recognition (CVPR)*, 2017, pp. 2117-2125.
        https://arxiv.org/abs/1612.03144
    """

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 2,
        scale_channels: int = 32,
        scales: Sequence[int] = (1, 2, 4, 8, 16),
    ) -> None:
        super().__init__()
        self.scales = list(scales)
        self.encoders = nn.ModuleList(
            _ScaleEncoder(in_channels, scale_channels) for _ in scales
        )
        feat_dim = scale_channels * len(scales)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for s, enc in zip(self.scales, self.encoders):
            x_s = F.avg_pool1d(x, kernel_size=s, stride=s) if s > 1 else x
            feats.append(enc(x_s))
        return self.classifier(torch.cat(feats, dim=1))
