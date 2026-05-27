import torch
import torch.nn as nn

from .resnet import ResBlock1D


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
    """Amplitude-Phase-Frequency decoupled stream encoder with attention fusion.

    Expects 4-channel 'apf' input: [log_mag, cos_phase, sin_phase, inst_freq].
    Three independent StreamEncoders process the amplitude, phase (cos/sin), and
    instantaneous-frequency channels separately, then the resulting 64-d feature
    vectors are fused via a single multi-head self-attention layer. The attention
    mechanism learns to weight the three streams per-example without requiring the
    network to discover the APF decomposition from raw I/Q.

    cos/sin phase encoding eliminates the +/-pi branch-cut discontinuity present in
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
        tokens = torch.stack([amp, phase, freq], dim=1)
        out, _ = self.attn(tokens, tokens, tokens)
        out = self.norm(out + tokens)
        return self.classifier(out.mean(dim=1))


class MultiStreamNet(nn.Module):
    """Channel-wise multi-stream encoder with attention fusion.

    Expects arbitrary C-channel time-series input. Each channel is processed by
    an independent StreamEncoder, then the resulting feature tokens are fused via
    a single multi-head self-attention layer. The number of attention heads is
    derived from the input channel count, allowing channel-independent views to
    interact after per-channel feature extraction.

    This tests late fusion of independent channel views against the early fusion
    used by standard convolutional backbones.

    The channel-token layout is related to inverted time-series Transformers,
    while the late-fusion comparison follows prior multi-channel AMC work.

    Citations:
        Liu, Yong, et al. "iTransformer: Inverted Transformers are Effective for
        Time Series Forecasting." *International Conference on Learning
        Representations (ICLR)*, 2024.
        https://arxiv.org/abs/2310.06625

        Zhang, Z., et al. "Multi-channel Fusion Convolutional Neural Networks for
        Automatic Modulation Classification." *IEEE Access*, vol. 8, 2020.
        https://doi.org/10.1109/ACCESS.2020.2982633

        Ramjee, Subramanian, et al. "Fast Deep Learning for Automatic Modulation
        Classification." *IEEE Access*, vol. 7, 2019.
        https://doi.org/10.1109/ACCESS.2019.2916568
    """

    def __init__(self, n_classes: int, in_channels: int, stream_dim: int = 64) -> None:
        super().__init__()
        self.num_heads = in_channels
        self.stream_dim = ((stream_dim + in_channels - 1) // in_channels) * in_channels
        self.streams = nn.ModuleList([
            _StreamEncoder(1, self.stream_dim) for _ in range(in_channels)
        ])
        self.attn = nn.MultiheadAttention(
            embed_dim=self.stream_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.stream_dim)
        self.classifier = nn.Linear(self.stream_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([
            enc(x[:, i : i + 1]) for i, enc in enumerate(self.streams)
        ], dim=1)
        out, _ = self.attn(tokens, tokens, tokens)
        out = self.norm(out + tokens)
        return self.classifier(out.mean(dim=1))
