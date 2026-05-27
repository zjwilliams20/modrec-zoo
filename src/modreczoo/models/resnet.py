from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ResNet2D(nn.Module):
    """2D ResNet with anisotropic kernels for image-like signal representations.

    Follows the residual block structure of He et al., adapted for 2D spectrogram
    inputs. Uses asymmetric (freq_kernel x time_kernel) convolutions throughout to
    reflect the different physical scales of frequency and time structure.

    Citation:
        He, Kaiming, et al. "Deep Residual Learning for Image Recognition."
        *Proceedings of the IEEE Conference on Computer Vision and Pattern
        Recognition (CVPR)*, 2016, pp. 770-778.
        https://arxiv.org/abs/1512.03385
    """

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
    """1D adaptation of the ResNet-18 architecture.

    Stem + four residual stages with progressive channel doubling and stride-2
    downsampling, followed by global average pooling. Residual shortcuts use a
    1x1 projection when channel counts change.

    Citation:
        He, Kaiming, et al. "Deep Residual Learning for Image Recognition."
        *Proceedings of the IEEE Conference on Computer Vision and Pattern
        Recognition (CVPR)*, 2016, pp. 770-778.
        https://arxiv.org/abs/1512.03385
    """

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
