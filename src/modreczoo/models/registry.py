from dataclasses import dataclass
from collections.abc import Callable

import torch.nn as nn

from .complex import ComplexCNN1D
from .cnn import CNN1D, CNN2D
from .dilated import DilatedCNN1D
from .mlp import FeatureMLP
from .multiscale import MultiScalePyramidNet
from .resnet import ResNet1D, ResNet2D
from .streams import APFNet, MultiStreamNet
from .transformer import PatchTransformer1D


@dataclass(frozen=True)
class ModelSpec:
    representation: str
    builder: Callable[..., nn.Module]
    required_channel_format: str | None = None


def _feature_mlp(n_classes: int, **_: object) -> nn.Module:
    return FeatureMLP(n_classes, 10)


def _cnn1d(n_classes: int, in_channels: int, **_: object) -> nn.Module:
    return CNN1D(n_classes, in_channels=in_channels)


def _resnet1d(n_classes: int, in_channels: int, **_: object) -> nn.Module:
    return ResNet1D(n_classes, in_channels=in_channels)


def _complex_cnn1d(n_classes: int, in_channels: int, **_: object) -> nn.Module:
    return ComplexCNN1D(n_classes, in_channels=in_channels)


def _dilated_cnn1d(n_classes: int, in_channels: int, **_: object) -> nn.Module:
    return DilatedCNN1D(n_classes, in_channels=in_channels)


def _multiscale_pyramid_1d(n_classes: int, in_channels: int, **_: object) -> nn.Module:
    return MultiScalePyramidNet(n_classes, in_channels=in_channels)


def _multi_stream_1d(n_classes: int, in_channels: int, **_: object) -> nn.Module:
    return MultiStreamNet(n_classes, in_channels=in_channels)


def _apf_net_1d(n_classes: int, in_channels: int, **_: object) -> nn.Module:
    return APFNet(n_classes, in_channels=in_channels)


def _csp_canonical_mlp(n_classes: int, **_: object) -> nn.Module:
    from modreczoo.features import N_CSP_CANONICAL_FEATURES

    return FeatureMLP(n_classes, N_CSP_CANONICAL_FEATURES)


def _csp_expert_mlp(n_classes: int, **_: object) -> nn.Module:
    from modreczoo.features import N_CSP_EXPERT_FEATURES

    return FeatureMLP(n_classes, N_CSP_EXPERT_FEATURES)


def _spectrogram_cnn(
    n_classes: int,
    in_channels: int,
    spectrogram_base_channels: int,
    **_: object,
) -> nn.Module:
    return CNN2D(
        n_classes,
        in_channels=in_channels,
        base_channels=spectrogram_base_channels,
    )


def _spectrogram_resnet(
    n_classes: int,
    in_channels: int,
    spectrogram_base_channels: int,
    spectrogram_freq_kernel: int,
    spectrogram_time_kernel: int,
    **_: object,
) -> nn.Module:
    return ResNet2D(
        n_classes,
        in_channels=in_channels,
        base_channels=spectrogram_base_channels,
        freq_kernel=spectrogram_freq_kernel,
        time_kernel=spectrogram_time_kernel,
    )


def _patch_transformer_1d(
    n_classes: int,
    in_channels: int,
    n_samples: int,
    transformer_patch_size: int,
    transformer_d_model: int,
    transformer_n_heads: int,
    transformer_n_layers: int,
    **_: object,
) -> nn.Module:
    return PatchTransformer1D(
        n_classes,
        in_channels=in_channels,
        n_samples=n_samples,
        patch_size=transformer_patch_size,
        d_model=transformer_d_model,
        n_heads=transformer_n_heads,
        n_layers=transformer_n_layers,
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "time_cnn": ModelSpec("time", _cnn1d),
    "frequency_cnn": ModelSpec("frequency", _cnn1d),
    "spectrogram_cnn": ModelSpec("spectrogram", _spectrogram_cnn),
    "spectrogram_resnet": ModelSpec("spectrogram", _spectrogram_resnet),
    "iq_features_mlp": ModelSpec("iq_features", _feature_mlp),
    "resnet_1d": ModelSpec("time", _resnet1d),
    "complex_cnn_1d": ModelSpec(
        "time",
        _complex_cnn1d,
        required_channel_format="real_imag",
    ),
    "dilated_cnn_1d": ModelSpec("time", _dilated_cnn1d),
    "patch_transformer_1d": ModelSpec("time", _patch_transformer_1d),
    "multiscale_pyramid_1d": ModelSpec("time", _multiscale_pyramid_1d),
    "multi_stream_1d": ModelSpec("time", _multi_stream_1d),
    "apf_net_1d": ModelSpec(
        "time",
        _apf_net_1d,
        required_channel_format="apf",
    ),
    "multilag_net_1d": ModelSpec(
        "time",
        _resnet1d,
        required_channel_format="multilag",
    ),
    "cyclic_caf_1d": ModelSpec(
        "time",
        _resnet1d,
        required_channel_format="cyclic_caf",
    ),
    "scf_resnet": ModelSpec(
        "spectrogram",
        _spectrogram_resnet,
        required_channel_format="scf",
    ),
    "csp_canonical_mlp": ModelSpec("csp_canonical", _csp_canonical_mlp),
    "csp_expert_mlp": ModelSpec("csp_features", _csp_expert_mlp),
}

MODEL_NAMES = tuple(MODEL_SPECS)
MODEL_REPRESENTATIONS = {
    model_name: spec.representation for model_name, spec in MODEL_SPECS.items()
}
MODEL_REQUIRED_CHANNEL_FORMATS: dict[str, str] = {
    model_name: channel_format
    for model_name, spec in MODEL_SPECS.items()
    if (channel_format := spec.required_channel_format) is not None
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
) -> tuple[nn.Module, str]:
    try:
        spec = MODEL_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc

    model = spec.builder(
        n_classes=n_classes,
        n_samples=n_samples,
        in_channels=in_channels,
        spectrogram_base_channels=spectrogram_base_channels,
        spectrogram_freq_kernel=spectrogram_freq_kernel,
        spectrogram_time_kernel=spectrogram_time_kernel,
        transformer_patch_size=transformer_patch_size,
        transformer_d_model=transformer_d_model,
        transformer_n_heads=transformer_n_heads,
        transformer_n_layers=transformer_n_layers,
    )
    return model, spec.representation
