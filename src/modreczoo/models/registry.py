from typing import Tuple

import torch.nn as nn

from .baselines import FeatureMLP, ResNet1D, SpectrogramCNN, TimeCNN
from .complex import ComplexCNN1D
from .dilated import DilatedCNN1D


MODEL_REPRESENTATIONS = {
    "time_cnn": "time",
    "frequency_cnn": "frequency",
    "spectrogram_cnn": "spectrogram",
    "feature_mlp": "features",
    "resnet_1d": "time",
    "complex_cnn_1d": "time",
    "dilated_cnn_1d": "time",
}


def representation_for_model(model_name: str) -> str:
    try:
        return MODEL_REPRESENTATIONS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc


def make_model(model_name: str, n_classes: int, n_samples: int, in_channels: int = 2) -> Tuple[nn.Module, str]:
    if model_name == "time_cnn":
        return TimeCNN(n_classes, n_samples, in_channels=in_channels), representation_for_model(model_name)
    if model_name == "frequency_cnn":
        return TimeCNN(n_classes, n_samples, in_channels=in_channels), representation_for_model(model_name)
    if model_name == "spectrogram_cnn":
        return SpectrogramCNN(n_classes, in_channels=in_channels), representation_for_model(model_name)
    if model_name == "feature_mlp":
        return FeatureMLP(n_classes, 10), representation_for_model(model_name)
    if model_name == "resnet_1d":
        return ResNet1D(n_classes, in_channels=in_channels), representation_for_model(model_name)
    if model_name == "complex_cnn_1d":
        return ComplexCNN1D(n_classes, in_channels=in_channels), representation_for_model(model_name)
    if model_name == "dilated_cnn_1d":
        return DilatedCNN1D(n_classes, in_channels=in_channels), representation_for_model(model_name)
    raise ValueError(f"Unsupported model: {model_name}")
