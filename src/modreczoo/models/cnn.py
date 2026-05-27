import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Three-layer 1D CNN baseline (VT-CNN2 architecture).

    Architecture follows the VT-CNN2 design introduced for automatic modulation
    recognition on over-the-air signals.

    Citation:
        O'Shea, Timothy J., Johnathan Corgan, and T. Charles Clancy. "Convolutional
        Radio Modulation Recognition Networks." *International Conference on
        Engineering Applications of Neural Networks (EANN)*, 2016.
        https://arxiv.org/abs/1602.04105
    """

    def __init__(self, n_classes: int, in_channels: int = 2) -> None:
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


class CNN2D(nn.Module):
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
