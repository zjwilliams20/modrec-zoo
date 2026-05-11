from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import ResBlock1D, ResNet1D


class PatchTransformer1D(nn.Module):
    """ViT-style patch transformer adapted for 1D I/Q signals.

    Divides the signal into non-overlapping patches, linearly embeds each patch,
    prepends a learnable CLS token, adds learnable positional encodings, and
    encodes with a stack of pre-norm TransformerEncoder layers. Classification
    reads from the CLS token.

    Follows the ViT architecture directly, substituting 1D patch flattening for
    2D image patch extraction. Attention mechanism from the original Transformer.

    Citations:
        Dosovitskiy, Alexey, et al. "An Image Is Worth 16x16 Words: Transformers
        for Image Recognition at Scale." *International Conference on Learning
        Representations (ICLR)*, 2021.
        https://arxiv.org/abs/2010.11929

        Vaswani, Ashish, et al. "Attention Is All You Need." *Advances in Neural
        Information Processing Systems (NeurIPS)*, vol. 30, 2017.
        https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 2,
        n_samples: int = 2048,
        patch_size: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        n_patches = n_samples // patch_size
        patch_dim = in_channels * patch_size
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        P = self.patch_size
        n_patches = T // P
        # (B, C, T) → (B, n_patches, C*P)
        x = x[:, :, : n_patches * P].reshape(B, C, n_patches, P)
        x = x.permute(0, 2, 1, 3).reshape(B, n_patches, C * P)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, : n_patches + 1]
        x = self.transformer(x)
        return self.classifier(self.norm(x[:, 0]))


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
    stride 1×, 2×, 4×, 8×, 16×). At scale k the downsampled branch operates at
    approximately one sample per symbol when the true OSR is k, giving each branch
    a different effective symbol-rate hypothesis. Independent ScaleEncoders produce
    fixed-size feature vectors that are concatenated and classified.

    The parallel multi-scale pyramid structure is inspired by feature pyramid
    networks for multi-resolution representation.

    Citation:
        Lin, Tsung-Yi, et al. "Feature Pyramid Networks for Object Detection."
        *Proceedings of the IEEE Conference on Computer Vision and Pattern
        Recognition (CVPR)*, 2017, pp. 2117–2125.
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


class _StreamEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 64) -> None:
        super().__init__()
        c = 32
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            ResBlock1D(c, c * 2, stride=2),
            ResBlock1D(c * 2, c * 4, stride=2),
            ResBlock1D(c * 4, out_dim, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(dim=-1)


class APFNet(nn.Module):
    """Amplitude–Phase–Frequency decoupled stream encoder with attention fusion.

    Expects 4-channel 'apf' input: [log_mag, cos_phase, sin_phase, inst_freq].
    Three independent StreamEncoders process the amplitude, phase (cos/sin), and
    instantaneous-frequency channels separately, then the resulting 64-d feature
    vectors are fused via a single multi-head self-attention layer. The attention
    mechanism learns to weight the three streams per-example without requiring the
    network to discover the APF decomposition from raw I/Q.

    cos/sin phase encoding eliminates the ±π branch-cut discontinuity present in
    raw unwrapped phase.

    The cross-stream attention fusion follows the self-attention formulation of:
        Vaswani, Ashish, et al. "Attention Is All You Need." *Advances in Neural
        Information Processing Systems (NeurIPS)*, vol. 30, 2017.
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, n_classes: int, in_channels: int = 4, stream_dim: int = 64) -> None:
        super().__init__()
        if in_channels != 4:
            raise ValueError("APFNet requires 4-channel 'apf' input.")
        self.amp_enc = _StreamEncoder(1, stream_dim)
        self.phase_enc = _StreamEncoder(2, stream_dim)
        self.freq_enc = _StreamEncoder(1, stream_dim)
        self.attn = nn.MultiheadAttention(embed_dim=stream_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(stream_dim)
        self.classifier = nn.Linear(stream_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amp = self.amp_enc(x[:, 0:1])
        phase = self.phase_enc(x[:, 1:3])
        freq = self.freq_enc(x[:, 3:4])
        tokens = torch.stack([amp, phase, freq], dim=1)  # (B, 3, stream_dim)
        out, _ = self.attn(tokens, tokens, tokens)
        out = self.norm(out + tokens)
        return self.classifier(out.mean(dim=1))


class MultiLagNet(nn.Module):
    """Multi-lag conjugate product encoder for cyclostationary feature extraction.

    Extends the differential-complex approach (lag=1) to multiple lags. For each lag τ,
    computes z[n]·z*[n−τ] whose angle is the phase change over τ samples. Three lags
    (1, 4, 16) span short-, medium-, and long-range inter-symbol correlation across the
    OSR range of 1–20 without requiring prior knowledge of the symbol rate. The six
    resulting channels (2 per lag) are fed into a ResNet1D backbone.

    Citation:
        Gardner, William A. "Exploitation of Spectral Redundancy in Cyclostationary
        Signals." *IEEE Signal Processing Magazine*, vol. 8, no. 2, 1991, pp. 14–36.
        https://doi.org/10.1109/79.81007
    """

    _LAGS = (1, 4, 16)

    def __init__(self, n_classes: int, in_channels: int = 2) -> None:
        super().__init__()
        if in_channels != 2:
            raise ValueError("MultiLagNet requires real_imag input with exactly 2 channels.")
        self.backbone = ResNet1D(n_classes, in_channels=len(self._LAGS) * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, N) — real_imag
        z = torch.view_as_complex(x.permute(0, 2, 1).contiguous())  # (B, N) complex
        channels = []
        eps = torch.finfo(x.dtype).eps
        for tau in self._LAGS:
            delayed = torch.roll(z, tau, dims=-1)
            delayed[..., :tau] = 0
            prod = z * delayed.conj()  # (B, N)
            scale = prod.abs().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)
            channels.append(prod.real / scale)
            channels.append(prod.imag / scale)
        return self.backbone(torch.stack(channels, dim=1))


class CyclicCAFNet(nn.Module):
    """Cyclic Autocorrelation Function (CAF) spectrum classifier.

    For each lag τ, computes the DFT of z[n]·z*[n−τ] over all n, giving the
    CAF magnitude spectrum R^α(τ) indexed by cyclic frequency α=k/N. Cyclostationary
    signals exhibit peaks in |R^α| at α = k·f_sym; an unknown OSR shifts the peak
    position but leaves a detectable ridge. Three lags (1, 4, 16) produce three
    magnitude spectra (channels), fed into a ResNet1D backbone.

    Citations:
        Gardner, William A. "Exploitation of Spectral Redundancy in Cyclostationary
        Signals." *IEEE Signal Processing Magazine*, vol. 8, no. 2, 1991, pp. 14–36.
        https://doi.org/10.1109/79.81007
    """

    _LAGS = (1, 4, 16)

    def __init__(self, n_classes: int, in_channels: int = 2) -> None:
        super().__init__()
        if in_channels != 2:
            raise ValueError("CyclicCAFNet requires real_imag input with exactly 2 channels.")
        self.backbone = ResNet1D(n_classes, in_channels=len(self._LAGS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, N) — real_imag
        z = torch.view_as_complex(x.permute(0, 2, 1).contiguous())  # (B, N) complex
        eps = torch.finfo(x.dtype).eps
        spectra = []
        for tau in self._LAGS:
            delayed = torch.roll(z, tau, dims=-1)
            delayed[..., :tau] = 0
            prod = z * delayed.conj()  # (B, N) complex
            r_alpha = torch.fft.fft(prod, dim=-1).abs()  # (B, N) magnitude spectrum
            scale = r_alpha.amax(dim=-1, keepdim=True).clamp(min=eps)
            spectra.append(r_alpha / scale)
        return self.backbone(torch.stack(spectra, dim=1))
