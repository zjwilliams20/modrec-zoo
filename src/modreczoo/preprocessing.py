import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PREPROCESSOR_NAMES = ("none", "normalize", "learned_fir", "radio_transform")


class IQNormalize(nn.Module):
    """Differentiable per-example centering and RMS normalization."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("IQNormalize expects input shaped (batch, channels, time).")
        x = x - x.mean(dim=-1, keepdim=True)
        power = x.square().sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
        scale = torch.sqrt(power.clamp_min(self.eps))
        return x / scale


class LearnedFIRPreprocessor(nn.Module):
    """Small learnable FIR frontend initialized to the identity map.

    This is intentionally a light drop-in frontend: it follows the learnable
    filterbank direction used by SincNet/LEAF-style audio frontends while keeping
    the default transform behavior equivalent at initialization.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 31) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("learned_fir kernel size must be a positive odd integer.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self._init_identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("LearnedFIRPreprocessor expects input shaped (batch, channels, time).")
        return self.conv(x)

    def _init_identity(self) -> None:
        with torch.no_grad():
            self.conv.weight.zero_()
            center = self.kernel_size // 2
            for channel in range(min(self.in_channels, self.out_channels)):
                self.conv.weight[channel, channel, center] = 1.0


class RadioTransformPreprocessor(nn.Module):
    """RTN/STN-style differentiable synchronizer for real/imaginary I/Q.

    A small localization network predicts fractional time shift, carrier
    frequency offset, and carrier phase correction. The final layer is
    zero-initialized, so the module starts as an identity transform.
    """

    def __init__(
        self,
        max_time_shift: float = 8.0,
        max_frequency_shift: float = 0.02,
        max_phase_shift: float = math.pi,
    ) -> None:
        super().__init__()
        self.max_time_shift = max_time_shift
        self.max_frequency_shift = max_frequency_shift
        self.max_phase_shift = max_phase_shift
        self.localizer = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 3),
        )
        final = self.localizer[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError("RadioTransformPreprocessor expects real_imag input shaped (batch, 2, time).")
        params = torch.tanh(self.localizer(x))
        time_shift = params[:, 0] * self.max_time_shift
        frequency_shift = params[:, 1] * self.max_frequency_shift
        phase_shift = params[:, 2] * self.max_phase_shift
        x = self._fractional_shift(x, time_shift)
        return self._rotate(x, frequency_shift, phase_shift)

    def _fractional_shift(self, x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        batch, _, n_samples = x.shape
        if n_samples <= 1:
            return x
        base = torch.linspace(-1.0, 1.0, n_samples, device=x.device, dtype=x.dtype)
        base = base.view(1, 1, n_samples).expand(batch, 1, n_samples)
        delta = 2.0 * shift.view(batch, 1, 1).to(dtype=x.dtype) / float(n_samples - 1)
        grid_x = base - delta
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return F.grid_sample(
            x.unsqueeze(2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(2)

    def _rotate(self, x: torch.Tensor, frequency_shift: torch.Tensor, phase_shift: torch.Tensor) -> torch.Tensor:
        _, _, n_samples = x.shape
        t = torch.arange(n_samples, device=x.device, dtype=x.dtype).view(1, 1, n_samples)
        t = t - (n_samples - 1) / 2.0
        phase = 2.0 * math.pi * frequency_shift.view(-1, 1, 1).to(dtype=x.dtype) * t
        phase = phase + phase_shift.view(-1, 1, 1).to(dtype=x.dtype)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        real = x[:, 0:1]
        imag = x[:, 1:2]
        return torch.cat((real * cos_phase - imag * sin_phase, real * sin_phase + imag * cos_phase), dim=1)


def make_preprocessor(
    name: str,
    in_channels: int,
    out_channels: int | None = None,
    kernel_size: int = 31,
    max_time_shift: float = 8.0,
    max_frequency_shift: float = 0.02,
    max_phase_shift: float = math.pi,
) -> tuple[nn.Module | None, int]:
    if name == "none":
        return None, in_channels
    if name == "normalize":
        return IQNormalize(), in_channels
    if name == "learned_fir":
        channels = out_channels if out_channels is not None else in_channels
        return LearnedFIRPreprocessor(in_channels, channels, kernel_size=kernel_size), channels
    if name == "radio_transform":
        if in_channels != 2:
            raise ValueError("radio_transform requires real_imag input with exactly 2 channels.")
        return (
            RadioTransformPreprocessor(
                max_time_shift=max_time_shift,
                max_frequency_shift=max_frequency_shift,
                max_phase_shift=max_phase_shift,
            ),
            2,
        )
    raise ValueError(f"Unsupported differentiable preprocessor: {name}.")
