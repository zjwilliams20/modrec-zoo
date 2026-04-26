from typing import Tuple

import torch
import torch.nn as nn


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
    def __init__(self, n_classes: int, in_channels: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(96, n_classes)

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


def make_model(model_name: str, n_classes: int, n_samples: int) -> Tuple[nn.Module, str]:
    if model_name == "time_cnn":
        return TimeCNN(n_classes, n_samples), "time"
    if model_name == "frequency_cnn":
        return TimeCNN(n_classes, n_samples), "frequency"
    if model_name == "spectrogram_cnn":
        return SpectrogramCNN(n_classes), "spectrogram"
    if model_name == "feature_mlp":
        return FeatureMLP(n_classes, 10), "features"
    raise ValueError(f"Unsupported model: {model_name}")
