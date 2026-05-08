from typing import Sequence, Tuple

import torch.nn as nn

from .advanced import APFNet, MultiScalePyramidNet, PatchTransformer1D
from .baselines import FeatureMLP, ResNet1D, SpectrogramCNN, SpectrogramResNet, TimeCNN
from .complex import ComplexCNN1D
from .dilated import DilatedCNN1D


# Models that require a specific channel format regardless of --channel-format.
MODEL_REQUIRED_CHANNEL_FORMATS: dict[str, str] = {
    "apf_net_1d": "apf",
    "diff_resnet_1d": "differential_complex",
}

MODEL_REPRESENTATIONS = {
    "time_cnn": "time",
    "frequency_cnn": "frequency",
    "spectrogram_cnn": "spectrogram",
    "spectrogram_resnet": "spectrogram",
    "feature_mlp": "features",
    "resnet_1d": "time",
    "complex_cnn_1d": "time",
    "dilated_cnn_1d": "time",
    "patch_transformer_1d": "time",
    "multiscale_pyramid_1d": "time",
    "diff_resnet_1d": "time",
    "apf_net_1d": "time",
}


def required_channel_format_for(model_name: str) -> str | None:
    return MODEL_REQUIRED_CHANNEL_FORMATS.get(model_name)


def representation_for_model(model_name: str) -> str:
    try:
        return MODEL_REPRESENTATIONS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc


def make_model(
    model_name: str,
    n_classes: int,
    n_samples: int,
    in_channels: int = 2,
    spectrogram_base_channels: int = 24,
    spectrogram_freq_kernel: int = 5,
    spectrogram_time_kernel: int = 3,
    transformer_patch_size: int = 32,
    transformer_d_model: int = 128,
    transformer_n_heads: int = 4,
    transformer_n_layers: int = 4,
) -> Tuple[nn.Module, str]:
    if model_name == "time_cnn":
        return TimeCNN(n_classes, n_samples, in_channels=in_channels), representation_for_model(model_name)
    if model_name == "frequency_cnn":
        return TimeCNN(n_classes, n_samples, in_channels=in_channels), representation_for_model(model_name)
    if model_name == "spectrogram_cnn":
        return (
            SpectrogramCNN(
                n_classes,
                in_channels=in_channels,
                base_channels=spectrogram_base_channels,
            ),
            representation_for_model(model_name),
        )
    if model_name == "spectrogram_resnet":
        return (
            SpectrogramResNet(
                n_classes,
                in_channels=in_channels,
                base_channels=spectrogram_base_channels,
                freq_kernel=spectrogram_freq_kernel,
                time_kernel=spectrogram_time_kernel,
            ),
            representation_for_model(model_name),
        )
    if model_name == "feature_mlp":
        return FeatureMLP(n_classes, 10), representation_for_model(model_name)
    if model_name == "resnet_1d":
        return ResNet1D(n_classes, in_channels=in_channels), representation_for_model(model_name)
    if model_name == "complex_cnn_1d":
        return ComplexCNN1D(n_classes, in_channels=in_channels), representation_for_model(model_name)
    if model_name == "dilated_cnn_1d":
        return DilatedCNN1D(n_classes, in_channels=in_channels), representation_for_model(model_name)
    if model_name == "patch_transformer_1d":
        return (
            PatchTransformer1D(
                n_classes,
                in_channels=in_channels,
                n_samples=n_samples,
                patch_size=transformer_patch_size,
                d_model=transformer_d_model,
                n_heads=transformer_n_heads,
                n_layers=transformer_n_layers,
            ),
            representation_for_model(model_name),
        )
    if model_name == "multiscale_pyramid_1d":
        return MultiScalePyramidNet(n_classes, in_channels=in_channels), representation_for_model(model_name)
    if model_name == "diff_resnet_1d":
        return ResNet1D(n_classes, in_channels=in_channels), representation_for_model(model_name)
    if model_name == "apf_net_1d":
        return APFNet(n_classes, in_channels=in_channels), representation_for_model(model_name)
    raise ValueError(f"Unsupported model: {model_name}")
