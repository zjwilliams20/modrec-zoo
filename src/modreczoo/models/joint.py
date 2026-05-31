"""Joint models combining CSP expert features with a raw-signal CNN branch.

Input representation: "joint_csp" — a (6 + N_CSP_EXPERT_FEATURES, N) tensor where:
  - Channels 0–5: complex_powers of the signal (Re/Im of z, z², z⁴)
  - Channels 6–112: 107 CSP expert features broadcast across the time axis

Five model variants:

JointCSPCNN:
  Signal branch → compact ResNet1D with Global Average Pooling → 256-d.
  Architecture equivalent to JointCSPCNN v2b (base_ch=32, LR=2e-3) that achieved
  87.12% (best single seed) on baseline_32768_200k — vs 80.12% CSP-expert-MLP alone.

JointCSPAttn:
  Signal branch → compact ResNet1D with Attention Pooling → 256-d.
  Replaces GAP with a learned temporal attention mask: each of the T time steps
  after the ResNet stages is scored by a small MLP, softmax-normalized, and
  used as a weighted average. Confirmed WORSE than GAP (86.22% vs 86.76% s0):
  cyclostationary statistics are globally uniform, so every time step contributes
  equally to E[d⁴ₜ] — GAP is theoretically optimal pooling.

JointCSPDual:
  Signal branch with 12-channel input: complex_powers (6ch) + unit_phasor_powers (6ch).
  unit_phasor_powers is computed from complex_powers in the forward pass — no data
  pipeline changes required. Theoretically motivated by complementarity:
  - complex_powers: retains RMS(z²) = amplitude kurtosis → separates QAM from PSK
  - unit_phasor_powers: u=z/|z|; u⁴ autocorrelation profile → discriminates PSK order
  The ResNet can allocate early filters to amplitude variation, late filters to phase.

JointCSPFiLM:
  CSP branch runs first → produces a conditioning vector → Feature-wise Linear
  Modulation applied at each ResNet stage in the signal branch via (1+γ)·h + β.
  Uses delta-form FiLM (γ≈0, β≈0 at init) so training starts as a standard ResNet
  and gradually learns to modulate signal features based on the CSP context.
  Unlike the concatenation-only approach, the signal CNN sees the CSP verdict
  *during* feature extraction, not just at the final classification head.

JointCSPDilated:
  Signal branch → DilatedCNN backbone (6 cells, d=1,2,4,8,16,32) with multi-scale
  avg+max pooling → 384-d embedding.  The 6 temporal scales are structurally
  complementary to the CSP global moment summaries: CSP integrates over the full
  signal while each dilated cell pools at a specific temporal scale.  This reduces
  redundancy vs. JointCSPCNN where ResNet's GAP is another global-averaging
  operation similar to the CSP cumulant estimators.
  ~570k params (vs 900k for JointCSPCNN) — the dilated backbone is 22× smaller
  than the ResNet signal branch yet captures multi-scale temporal structure.
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
        return self.forward_parts(sig, csp)

    def forward_parts(self, complex_powers: torch.Tensor, csp_raw: torch.Tensor) -> torch.Tensor:
        """Efficient entry point for training scripts that keep tensors split.

        Avoids broadcasting the CSP vector across the full time axis:
        ``complex_powers`` (B, 6, N) + ``csp_raw`` (B, N_CSP) use only 786 KB +
        428 B per sample vs. 14.8 MB for the broadcast joint tensor.

        Args:
            complex_powers: (B, 6, N) pre-computed complex power channels.
            csp_raw: (B, N_CSP) z-score normalised CSP feature vector.
        """
        sig_emb = self.sig_branch(complex_powers)
        csp_emb = self.csp_branch(csp_raw)
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
        return self.forward_parts(sig, csp)

    def forward_parts(self, complex_powers: torch.Tensor, csp_raw: torch.Tensor) -> torch.Tensor:
        """Override: compute unit_phasor_powers on-the-fly before the 12-ch branch.

        Args:
            complex_powers: (B, 6, N) complex power channels.
            csp_raw: (B, N_CSP) z-score normalised CSP features.
        """
        upow  = self._unit_phasor_powers(complex_powers)    # (B, 6, N)
        sig12 = torch.cat([complex_powers, upow], dim=1)    # (B, 12, N)
        sig_emb = self.sig_branch(sig12)
        csp_emb = self.csp_branch(csp_raw)
        return self.head(torch.cat([sig_emb, csp_emb], dim=1))


# ── FiLM building blocks ──────────────────────────────────────────────────────

class _FiLMResBlock1D(nn.Module):
    """Residual block with Feature-wise Linear Modulation (FiLM) conditioning.

    Applies delta-form FiLM after the second conv+BN: ``(1 + γ) · h + β``.
    Delta form: the FiLM generator's weights/biases default to small values, so
    γ ≈ 0 and β ≈ 0 at initialisation — the block behaves like a plain ResBlock
    until the CSP gradient signal trains the conditioning.

    Args:
        ci: input channels.
        co: output channels.
        s: stride (default 1).
        cond_dim: dimension of the conditioning vector (default 256).
    """

    def __init__(self, ci: int, co: int, s: int = 1, cond_dim: int = 256) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(ci, co, 3, stride=s, padding=1, bias=False)
        self.b1 = nn.BatchNorm1d(co)
        self.c2 = nn.Conv1d(co, co, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm1d(co)
        self.skip = (
            nn.Sequential(nn.Conv1d(ci, co, 1, stride=s, bias=False), nn.BatchNorm1d(co))
            if s != 1 or ci != co else nn.Identity()
        )
        self.film = nn.Linear(cond_dim, 2 * co)   # generates (γ, β) per channel

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))   # (B, co, T)
        γβ = self.film(cond)                                  # (B, 2*co)
        γ = γβ[:, : h.shape[1]].unsqueeze(-1)                # (B, co, 1)
        β = γβ[:, h.shape[1] :].unsqueeze(-1)                # (B, co, 1)
        return F.relu((1.0 + γ) * h + β + self.skip(x))      # delta-form FiLM + residual


class _FiLMSignalBranch(nn.Module):
    """Compact ResNet1D with FiLM conditioning at every residual stage.

    Identical structure to :class:`_SignalBranch` (same stem, same 4 stages,
    same channel progression, same GAP) except each residual block is a
    :class:`_FiLMResBlock1D` that accepts a CSP conditioning vector.

    The conditioning vector is the same at every stage — the blocks learn
    to use different projections of it for different levels of abstraction.

    Args:
        in_ch: input channels (always 6 for complex_powers).
        base_ch: base channel count; doubles through stages to ``base_ch * 8``.
        cond_dim: dimension of the conditioning vector from the CSP branch.
    """

    def __init__(self, in_ch: int = 6, base_ch: int = 32, cond_dim: int = 256) -> None:
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.block1 = _FiLMResBlock1D(c,     c,     s=1, cond_dim=cond_dim)
        self.block2 = _FiLMResBlock1D(c,     c * 2, s=2, cond_dim=cond_dim)
        self.block3 = _FiLMResBlock1D(c * 2, c * 4, s=2, cond_dim=cond_dim)
        self.block4 = _FiLMResBlock1D(c * 4, c * 8, s=2, cond_dim=cond_dim)
        self.out_dim = c * 8

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.block1(h, cond)
        h = self.block2(h, cond)
        h = self.block3(h, cond)
        h = self.block4(h, cond)
        return h.mean(-1)   # GAP → (B, out_dim)


# ── FiLM full model ───────────────────────────────────────────────────────────

class JointCSPFiLM(nn.Module):
    """CSP-conditioned signal CNN via Feature-wise Linear Modulation.

    Unlike :class:`JointCSPCNN` (which concatenates CSP and signal embeddings
    only at the final head), JointCSPFiLM feeds the CSP embedding into the
    signal branch at *every* residual stage via FiLM scale/shift parameters.

    Processing order:
      1. CSP branch: 107-dim features → 2-block ResMLP → 256-d ``csp_emb``.
      2. Signal branch: 6-ch complex_powers → stem → 4× FiLM-conditioned
         ResBlocks (each using ``csp_emb`` for γ, β) → GAP → 256-d ``sig_emb``.
      3. Head: concat(sig_emb, csp_emb) = 512-d → 256 → 128 → n_classes.

    The head still includes the CSP embedding so the classifier sees both the
    raw-signal features *and* the original CSP context.

    Parameter overhead vs. JointCSPCNN: 4 FiLM generators (256 → {64, 128, 256, 512})
    ≈ +246k parameters (~27% increase over the base ~900k).

    Args:
        n_classes: number of output classes.
        base_ch: base channel count for the signal branch (default 32).
        csp_hidden: hidden dim for the CSP branch ResMLP (default 256).
        n_csp: number of CSP features (default 107; must match the dataset).
        csp_blocks: residual blocks in the CSP branch (default 2).
        drop: dropout probability in CSP branch and head (default 0.25).
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
        self.csp_branch = _CSPBranch(n_csp=n_csp, hidden=csp_hidden, n_blocks=csp_blocks, drop=drop)
        self.sig_branch = _FiLMSignalBranch(in_ch=6, base_ch=base_ch, cond_dim=csp_hidden)

        fused = self.sig_branch.out_dim + csp_hidden   # 256 + 256 = 512
        self.head = nn.Sequential(
            nn.Linear(fused, fused // 2), nn.BatchNorm1d(fused // 2), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(fused // 2, fused // 4), nn.BatchNorm1d(fused // 4), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(fused // 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6 + N_CSP, N)
        sig = x[:, :6, :]          # (B, 6, N)   complex_powers
        csp = x[:, 6:, 0]          # (B, N_CSP)  CSP features
        return self.forward_parts(sig, csp)

    def forward_parts(self, complex_powers: torch.Tensor, csp_raw: torch.Tensor) -> torch.Tensor:
        """Efficient entry point: avoids broadcasting the CSP vector.

        Args:
            complex_powers: (B, 6, N) complex power channels.
            csp_raw: (B, N_CSP) z-score normalised CSP features.
        """
        csp_emb = self.csp_branch(csp_raw)                # (B, 256) — run CSP first
        sig_emb = self.sig_branch(complex_powers, csp_emb)  # (B, 256) — CSP conditions CNN
        return self.head(torch.cat([sig_emb, csp_emb], dim=1))


# ── Dilated multi-scale signal branch ────────────────────────────────────────

class _DilatedCell(nn.Module):
    """Single residual dilated-convolution cell (BN + ReLU + residual)."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)) + x)


class _DilatedSignalBranch(nn.Module):
    """DilatedCNN backbone producing a multi-scale pooled embedding for joint fusion.

    Same structure as ``DilatedCNN1D`` but without the classification head.
    After each dilated cell both avg- and max-pooled features are collected,
    yielding a ``2 × channels × len(dilations)``-dimensional descriptor that
    encodes structure at every temporal scale simultaneously.

    Args:
        in_ch: input channels (6 for complex_powers).
        channels: fixed channel width throughout all dilated cells (default 32).
        dilations: dilation factors for each cell (default (1,2,4,8,16,32)).
    """

    def __init__(
        self,
        in_ch: int = 6,
        channels: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.cells = nn.ModuleList(_DilatedCell(channels, d) for d in dilations)
        self.out_dim = 2 * channels * len(dilations)  # avg + max per cell

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        pooled: list[torch.Tensor] = []
        for cell in self.cells:
            h = cell(h)
            pooled.append(F.adaptive_avg_pool1d(h, 1).squeeze(-1))
            pooled.append(F.adaptive_max_pool1d(h, 1).squeeze(-1))
        return torch.cat(pooled, dim=1)   # (B, 2*channels*n_cells)


class JointCSPDilated(nn.Module):
    """CSP expert features fused with a DilatedCNN multi-scale signal embedding.

    The DilatedCNN signal branch pools avg+max features at every dilated cell
    (d=1,2,4,8,16,32), producing a 384-d multi-scale descriptor that captures
    temporal structure at six explicit scales.  This is structurally complementary
    to the CSP 256-d embedding, which integrates over the full signal — the two
    branches encode orthogonal views (global moments vs. multi-scale local structure).

    By contrast, JointCSPCNN's ResNet signal branch produces a single GAP 256-d
    vector, which is a global average similar to CSP's own moment computation.
    The dilated branch is expected to be less redundant with CSP and to add more
    complementary information per parameter.

    Parameter budget (~570k) is 37% smaller than JointCSPCNN (~900k) because the
    dilated backbone (22k params) is 20× smaller than the ResNet1D signal branch
    (≈440k) while providing richer scale coverage.

    Args:
        n_classes: number of output classes.
        channels: fixed channel width in the dilated backbone (default 32).
        csp_hidden: hidden dim for the CSP branch ResMLP (default 256).
        n_csp: number of CSP features (default 107; must match the dataset).
        csp_blocks: residual blocks in the CSP branch (default 2).
        drop: dropout probability in CSP branch and head (default 0.25).
    """

    def __init__(
        self,
        n_classes: int,
        channels: int = 32,
        csp_hidden: int = 256,
        n_csp: int = N_CSP,
        csp_blocks: int = 2,
        drop: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_csp = n_csp
        self.sig_branch = _DilatedSignalBranch(in_ch=6, channels=channels)
        self.csp_branch = _CSPBranch(n_csp=n_csp, hidden=csp_hidden, n_blocks=csp_blocks, drop=drop)

        fused = self.sig_branch.out_dim + csp_hidden   # 384 + 256 = 640
        self.head = nn.Sequential(
            nn.Linear(fused, fused // 2), nn.BatchNorm1d(fused // 2), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(fused // 2, fused // 4), nn.BatchNorm1d(fused // 4), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(fused // 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6 + N_CSP, N)
        sig = x[:, :6, :]          # (B, 6, N)   complex_powers
        csp = x[:, 6:, 0]          # (B, N_CSP)  CSP features
        return self.forward_parts(sig, csp)

    def forward_parts(self, complex_powers: torch.Tensor, csp_raw: torch.Tensor) -> torch.Tensor:
        """Efficient entry point: avoids broadcasting the CSP vector.

        Args:
            complex_powers: (B, 6, N) complex power channels.
            csp_raw: (B, N_CSP) z-score normalised CSP features.
        """
        sig_emb = self.sig_branch(complex_powers)           # (B, 384) multi-scale
        csp_emb = self.csp_branch(csp_raw)                  # (B, 256) global
        return self.head(torch.cat([sig_emb, csp_emb], dim=1))
