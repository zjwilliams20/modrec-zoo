from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ComplexTensor = Tuple[torch.Tensor, torch.Tensor]


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__()
        self.real = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.imag = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

    def forward(self, x: ComplexTensor) -> ComplexTensor:
        xr, xi = x
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        return yr, yi


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.real = nn.BatchNorm1d(channels)
        self.imag = nn.BatchNorm1d(channels)

    def forward(self, x: ComplexTensor) -> ComplexTensor:
        xr, xi = x
        return self.real(xr), self.imag(xi)


class ComplexCNNBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: bool) -> None:
        super().__init__()
        self.conv = ComplexConv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = ComplexBatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x: ComplexTensor) -> ComplexTensor:
        xr, xi = self.bn(self.conv(x))
        xr = self.pool(F.relu(xr))
        xi = self.pool(F.relu(xi))
        return xr, xi


class ComplexCNN1D(nn.Module):
    """Small complex-valued CNN baseline for real/imag time-domain I/Q.

    Implements complex-valued convolution as two tied real convolutions:
        y_r = W_r * x_r − W_i * x_i
        y_i = W_r * x_i + W_i * x_r
    preserving the algebraic structure of C throughout the feature hierarchy.

    Citation:
        Trabelsi, Chiheb, et al. "Deep Complex Networks." *International Conference
        on Learning Representations (ICLR)*, 2018.
        https://arxiv.org/abs/1705.09792
    """

    def __init__(self, n_classes: int, in_channels: int = 2) -> None:
        super().__init__()
        if in_channels != 2:
            raise ValueError("ComplexCNN1D requires real_imag input with exactly 2 channels.")
        self.net = nn.Sequential(
            ComplexCNNBlock1D(1, 32, kernel_size=9, pool=True),
            ComplexCNNBlock1D(32, 64, kernel_size=7, pool=True),
            ComplexCNNBlock1D(64, 128, kernel_size=5, pool=False),
        )
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = x[:, 0:1]
        xi = x[:, 1:2]
        xr, xi = self.net((xr, xi))
        xr = F.adaptive_avg_pool1d(xr, 1).squeeze(-1)
        xi = F.adaptive_avg_pool1d(xi, 1).squeeze(-1)
        return self.classifier(torch.cat((xr, xi), dim=1))
