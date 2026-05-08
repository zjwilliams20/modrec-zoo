from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeCNN(nn.Module):
    def __init__(self, n_classes: int, n_samples: int, in_channels: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.net(x).squeeze(-1))


class SpectrogramCNN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_channels: int = 2,
        base_channels: int = 24,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("SpectrogramCNN kernel_size must be a positive odd integer.")
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(c3, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.net(x).flatten(1))


class FeatureMLP(nn.Module):
    def __init__(self, n_classes: int, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: Tuple[int, int] = (3, 3),
    ) -> None:
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        skip = self.downsample(x) if self.downsample is not None else x
        return F.relu(out + skip)


class SpectrogramResNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_channels: int = 2,
        base_channels: int = 32,
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        freq_kernel: int = 5,
        time_kernel: int = 3,
    ) -> None:
        super().__init__()
        c = base_channels
        ks = (freq_kernel, time_kernel)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=ks, stride=2, padding=(freq_kernel // 2, time_kernel // 2), bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stages = nn.Sequential(
            self._make_stage(c, c, blocks_per_stage[0], stride=1, ks=ks),
            self._make_stage(c, c * 2, blocks_per_stage[1], stride=2, ks=ks),
            self._make_stage(c * 2, c * 4, blocks_per_stage[2], stride=2, ks=ks),
            self._make_stage(c * 4, c * 8, blocks_per_stage[3], stride=2, ks=ks),
        )
        self.classifier = nn.Linear(c * 8, n_classes)

    def _make_stage(
        self, in_channels: int, out_channels: int, n_blocks: int, stride: int, ks: Tuple[int, int]
    ) -> nn.Sequential:
        blocks: list[nn.Module] = [ResBlock2D(in_channels, out_channels, stride, kernel_size=ks)]
        for _ in range(1, n_blocks):
            blocks.append(ResBlock2D(out_channels, out_channels, kernel_size=ks))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stages(self.stem(x)).mean((-2, -1)))


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        skip = self.downsample(x) if self.downsample is not None else x
        return F.relu(out + skip)


class ResNet1D(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_channels: int = 2,
        base_channels: int = 32,
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stages = nn.Sequential(
            self._make_stage(c, c, blocks_per_stage[0], stride=1),
            self._make_stage(c, c * 2, blocks_per_stage[1], stride=2),
            self._make_stage(c * 2, c * 4, blocks_per_stage[2], stride=2),
            self._make_stage(c * 4, c * 8, blocks_per_stage[3], stride=2),
        )
        self.classifier = nn.Linear(c * 8, n_classes)

    def _make_stage(self, in_channels: int, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks: list[nn.Module] = [ResBlock1D(in_channels, out_channels, stride)]
        for _ in range(1, n_blocks):
            blocks.append(ResBlock1D(out_channels, out_channels))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stages(self.stem(x)).mean(-1))
