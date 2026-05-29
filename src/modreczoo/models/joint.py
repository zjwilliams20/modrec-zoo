"""Joint models combining CSP expert features with a raw-signal CNN branch.

Input representation: "joint_csp" — a (6 + N_CSP_EXPERT_FEATURES, N) tensor where:
  - Channels 0–5: complex_powers of the signal (Re/Im of z, z², z⁴)
  - Channels 6–112: 107 CSP expert features broadcast across the time axis

Three model variants:

JointCSPCNN:
  Signal branch → compact ResNet1D with Global Average Pooling → 256-d.
  Architecture equivalent to JointCSPCNN v2b (base_ch=32, LR=2e-3) that achieved
  81.25% on baseline_4096 and 81.22% 4-model ensemble on baseline_32768_40k.

JointCSPAttn:
  Signal branch → compact ResNet1D with Attention Pooling → 256-d.
  Replaces GAP with a learned temporal attention mask: each of the T time steps
  after the ResNet stages is scored by a small MLP, softmax-normalized, and
  used as a weighted average. Motivated by the temporal structure of AMC signals
  (pi/4-DQPSK phase transitions, symbol-boundary effects) and the empirical
  finding that GAP discards discriminative positional information for long signals.

JointCSPDual:
  Signal branch with 12-channel input: complex_powers (6ch) + unit_phasor_powers (6ch).
  unit_phasor_powers is computed from complex_powers in the forward pass — no data
  pipeline changes required. Theoretically motivated by complementarity:
  - complex_powers: retains RMS(z²) = amplitude kurtosis → separates QAM from PSK
  - unit_phasor_powers: u=z/|z|; u⁴ autocorrelation profile → discriminates PSK order
  The ResNet can allocate early filters to amplitude variation, late filters to phase.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CSP = 107   # must match modreczoo.features.N_CSP_EXPERT_FEATURES


# ── Building blocks ───────────────────────────────────────────────────────────

class _ResBlock1D(nn.Module):
    def __init__(self, ci: int, co: int, s: int = 1) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(ci, co, 3, stride=s, padding=1, bias=False)
        self.b1 = nn.BatchNorm1d(co)
        self.c2 = nn.Conv1d(co, co, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm1d(co)
        self.skip = (
            nn.Sequential(nn.Conv1d(ci, co, 1, stride=s, bias=False), nn.BatchNorm1d(co))
            if s != 1 or ci != co else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


class _ResBlockMLP(nn.Module):
    def __init__(self, d: int, drop: float = 0.25) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(),
            nn.Dropout(drop), nn.Linear(d, d), nn.BatchNorm1d(d),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ── Pooling variants ──────────────────────────────────────────────────────────

class _AttentionPool1D(nn.Module):
    """Additive attention pooling over the temporal axis.

    Scores each time step with a small 2-layer MLP, normalizes with softmax,
    and returns a weighted sum over the channel dimension. This allows the model
    to focus on the most class-discriminative time windows rather than treating
    all positions equally (as GAP does).

    Adds ~2 × (C × C//2 + C//2) parameters — for C=256 that is ~33k, roughly
    0.3% of the total JointCSPCNN parameter budget.

    Reference: Bahdanau et al. (2015) additive attention; Yang et al. (2016)
    hierarchical attention for document classification.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x_t = x.permute(0, 2, 1)                     # (B, T, C)
        attn = self.score(x_t).squeeze(-1)            # (B, T)
        attn = F.softmax(attn, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (x_t * attn).sum(1)                    # (B, C)


# ── Signal branch ─────────────────────────────────────────────────────────────

class _SignalBranch(nn.Module):
    """Compact ResNet1D over the complex_powers channels (6-ch input).

    Stem: Conv(k=7, s=2) + MaxPool(s=2) = 4× reduction.  Four stages of
    progressive channel doubling and stride-2 downsampling.  Final pooling
    is either Global Average Pooling (``pool='gap'``) or learned attention
    pooling (``pool='attn'``).  Produces a ``sig_dim``-dimensional embedding
    regardless of input length.

    Args:
        in_ch: input channels (always 6 for complex_powers).
        base_ch: base channel count; doubles through stages to ``base_ch * 8``.
            Default 32 → output dim 256.
        pool: ``'gap'`` (default) or ``'attn'`` (attention pooling).
    """

    def __init__(self, in_ch: int = 6, base_ch: int = 32, pool: str = "gap") -> None:
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.stages = nn.Sequential(
            _ResBlock1D(c,     c,     s=1),
            _ResBlock1D(c,     c * 2, s=2),
            _ResBlock1D(c * 2, c * 4, s=2),
            _ResBlock1D(c * 4, c * 8, s=2),
        )
        self.out_dim = c * 8
        if pool == "attn":
            self.pool: nn.Module = _AttentionPool1D(self.out_dim)
        else:
            self.pool = nn.Identity()   # GAP applied manually in forward

        self._use_attn = pool == "attn"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stages(self.stem(x))   # (B, out_dim, T)
        if self._use_attn:
            return self.pool(h)          # (B, out_dim)
        return h.mean(-1)               # GAP → (B, out_dim)


# ── CSP branch ────────────────────────────────────────────────────────────────

class _CSPBranch(nn.Module):
    """Two-block ResMLP over the 107-dim CSP expert feature vector.

    Args:
        n_csp: number of CSP features (default 107).
        hidden: hidden dim (default 256).
        n_blocks: number of residual blocks (default 2).
        drop: dropout probability (default 0.25).
    """

    def __init__(
        self,
        n_csp: int = N_CSP,
        hidden: int = 256,
        n_blocks: int = 2,
        drop: float = 0.25,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(n_csp, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.blocks = nn.Sequential(*[_ResBlockMLP(hidden, drop) for _ in range(n_blocks)])
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


# ── Full model ────────────────────────────────────────────────────────────────

class JointCSPCNN(nn.Module):
    """Two-branch AMC model: CSP expert features + raw signal CNN.

    Accepts the "joint_csp" representation: a (B, 6 + N_CSP, N) tensor where:
      - ``x[:, :6, :]``  — complex_powers channels (Re/Im of z, z², z⁴)
      - ``x[:, 6:, 0]``  — CSP expert features (same at every time step)

    The complex_powers channels do NOT have per-power RMS normalization applied
    (i.e. ``ri_norm`` is absent). z² and z⁴ retain their amplitude kurtosis
    information since z is already unit-power from ``normalize_signal()``.

    Args:
        n_classes: number of output classes.
        base_ch: base channel count for the signal branch (default 32).
        csp_hidden: hidden dim for the CSP branch ResMLP (default 256).
        n_csp: number of CSP features (default 107; must match the dataset).
        csp_blocks: number of ResMLP residual blocks in the CSP branch (default 2).
        drop: dropout probability in both branches and the fusion head (default 0.25).
    """

    def __init__(
        self,
        n_classes: int,
        base_ch: int = 32,
        csp_hidden: int = 256,
        n_csp: int = N_CSP,
        csp_blocks: int = 2,
        drop: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_csp = n_csp
        self.sig_branch = _SignalBranch(in_ch=6, base_ch=base_ch)
        self.csp_branch = _CSPBranch(n_csp=n_csp, hidden=csp_hidden, n_blocks=csp_blocks, drop=drop)

        fused = self.sig_branch.out_dim + csp_hidden
        self.head = nn.Sequential(
            nn.Linear(fused, fused // 2), nn.BatchNorm1d(fused // 2), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(fused // 2, fused // 4), nn.BatchNorm1d(fused // 4), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(fused // 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6 + N_CSP, N)
        sig  = x[:, :6, :]           # (B, 6, N)  — complex_powers
        csp  = x[:, 6:, 0]           # (B, N_CSP) — CSP features (constant across time)
        sig_emb = self.sig_branch(sig)
        csp_emb = self.csp_branch(csp)
        return self.head(torch.cat([sig_emb, csp_emb], dim=1))


class JointCSPAttn(JointCSPCNN):
    """JointCSPCNN with attention pooling in the signal branch instead of GAP.

    Identical architecture to JointCSPCNN except the signal branch uses
    ``_AttentionPool1D`` instead of Global Average Pooling. The attention
    mechanism learns to weight each temporal position by its discriminative
    contribution, which is especially useful for:
      - Long signals (32768 samples, ~2570 symbols): GAP dilutes local structure
      - pi/4-DQPSK vs 8PSK: differential phase transitions are position-dependent
      - QAM amplitude variation: certain symbol runs are more discriminative

    All other hyperparameters (base_ch, csp_hidden, csp_blocks, drop) are
    identical, making this a controlled comparison against JointCSPCNN.
    """

    def __init__(
        self,
        n_classes: int,
        base_ch: int = 32,
        csp_hidden: int = 256,
        n_csp: int = N_CSP,
        csp_blocks: int = 2,
        drop: float = 0.25,
    ) -> None:
        super().__init__(
            n_classes=n_classes,
            base_ch=base_ch,
            csp_hidden=csp_hidden,
            n_csp=n_csp,
            csp_blocks=csp_blocks,
            drop=drop,
        )
        # Replace the GAP signal branch with an attention-pooled one.
        # The output dimension is unchanged (base_ch * 8) so the fusion head
        # from the parent __init__ is still valid.
        self.sig_branch = _SignalBranch(in_ch=6, base_ch=base_ch, pool="attn")


class JointCSPDual(JointCSPCNN):
    """JointCSPCNN with 12-channel signal input: complex_powers + unit_phasor_powers.

    The ``unit_phasor_powers`` channels are computed on-the-fly from the first two
    complex_powers channels (Re(z), Im(z)) in the forward pass.  No data pipeline
    changes are required — the input tensor format is identical to JointCSPCNN.

    Theoretical motivation (complementary representations):
      - ``complex_powers`` [Ch 0–5]: Re/Im of z, z², z⁴.  z² and z⁴ retain their
        amplitude RMS, encoding amplitude kurtosis (QAM > PSK; key QAM/PSK separator).
      - ``unit_phasor_powers`` [Ch 6–11]: Re/Im of u², u⁴, u⁸ where u = z/(|z|+ε).
        u⁴ autocorrelation profile separates 4PSK (+1) / π/4-DQPSK (−1) / 8PSK (≈0).

    The ResNet sees both simultaneously (12-ch input); its filters can specialize:
    early filters can respond to amplitude variance (QAM/PSK separation), later
    filters to phase structure (PSK order discrimination).

    Output dimension and fusion head are unchanged relative to JointCSPCNN
    (base_ch * 8 from the signal branch + csp_hidden from the CSP branch).
    Parameter overhead: only the first Conv1d layer grows from 6→12 in_channels
    (+6 × 7 × base_ch weights ≈ +1344 params for base_ch=32 — negligible).
    """

    _EPS = 1e-7

    def __init__(
        self,
        n_classes: int,
        base_ch: int = 32,
        csp_hidden: int = 256,
        n_csp: int = N_CSP,
        csp_blocks: int = 2,
        drop: float = 0.25,
    ) -> None:
        super().__init__(
            n_classes=n_classes,
            base_ch=base_ch,
            csp_hidden=csp_hidden,
            n_csp=n_csp,
            csp_blocks=csp_blocks,
            drop=drop,
        )
        # Rebuild the signal branch to accept 12 channels instead of 6.
        # Output dimension (base_ch * 8) and the fusion head are unchanged.
        self.sig_branch = _SignalBranch(in_ch=12, base_ch=base_ch)

    @staticmethod
    def _unit_phasor_powers(sig: torch.Tensor) -> torch.Tensor:
        """Compute unit_phasor_powers from complex_powers channels on GPU.

        Args:
            sig: (B, 6, N) complex_powers tensor.  First 2 channels are Re(z), Im(z).
        Returns:
            (B, 6, N) unit_phasor_powers: [Re(u²), Im(u²), Re(u⁴), Im(u⁴), Re(u⁸), Im(u⁸)].
        """
        z_re = sig[:, 0:1, :]       # (B, 1, N)
        z_im = sig[:, 1:2, :]       # (B, 1, N)
        mag = torch.sqrt(z_re ** 2 + z_im ** 2) + JointCSPDual._EPS
        u_re = z_re / mag           # (B, 1, N) — unit phasor real part
        u_im = z_im / mag           # (B, 1, N) — unit phasor imag part
        channels = []
        for _ in range(3):          # powers 2, 4, 8 — computed by squaring iteratively
            # (u_re + j*u_im)^2 = u_re²−u_im², 2·u_re·u_im  (De Moivre-style)
            u_re_new = u_re ** 2 - u_im ** 2
            u_im_new = 2 * u_re * u_im
            u_re, u_im = u_re_new, u_im_new
            channels.extend([u_re, u_im])
        return torch.cat(channels, dim=1)   # (B, 6, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6 + N_CSP, N)
        sig = x[:, :6, :]                           # (B, 6, N) complex_powers
        csp = x[:, 6:, 0]                           # (B, N_CSP)
        upow = self._unit_phasor_powers(sig)         # (B, 6, N) unit_phasor_powers
        sig12 = torch.cat([sig, upow], dim=1)        # (B, 12, N)
        sig_emb = self.sig_branch(sig12)
        csp_emb = self.csp_branch(csp)
        return self.head(torch.cat([sig_emb, csp_emb], dim=1))
