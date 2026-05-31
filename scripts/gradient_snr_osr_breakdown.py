#!/usr/bin/env python3
"""
Gradient dominance breakdown by (SNR, col-axis) bin.

Per-sample cross-entropy loss proxies gradient magnitude contribution to
each update step: ||dL/dz||_2 = sqrt(2*(1-p_y)) is monotone in L = -log(p_y).

Trains a small model briefly (--epochs, default 10) OR loads a checkpoint,
then reports mean loss and fractional gradient weight per (SNR, col-axis) bin.

Default column axis is symbol_period (log-octave bins [2,4), [4,8), [8,16))
since that is the primary axis of variation in the default simulation setup.

Usage:
  # 1. Generate data with default params
  uv run modreczoo-simulate generate --output-dir data/breakdown \\
      --n-signals 4000 --channel awgn --sampler sobol --seed 0

  # 2. Run breakdown (trains briefly inside)
  uv run python scripts/gradient_snr_osr_breakdown.py --dataset-dir data/breakdown

  # 3. Use OSR as column axis instead
  uv run python scripts/gradient_snr_osr_breakdown.py \\
      --dataset-dir data/breakdown --col-axis osr

  # 4. With a pre-trained checkpoint (skip internal training)
  uv run python scripts/gradient_snr_osr_breakdown.py \\
      --dataset-dir data/breakdown --checkpoint path/to/model.pt

  # 5. Also compute exact per-sample gradient norms (slow)
  uv run python scripts/gradient_snr_osr_breakdown.py \\
      --dataset-dir data/breakdown --exact-grads --grad-subset 500
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from modreczoo.data import ModrecDataset, load_dataset, ordered_modulation_labels
from modreczoo.models import make_model, representation_for_model, required_channel_format_for
from modreczoo.training import SNR_BIN_WIDTH, input_channels_for


OSR_EDGES = [1.0, 2.0, 4.0, 8.0, 16.0, np.inf]
OSR_LABELS = ["[1,2)", "[2,4)", "[4,8)", "[8,16)", "[16,∞)"]

# Log-octave bins for symbol_period: [2,4), [4,8), [8,16)
SP_EDGES = [2, 4, 8, 16]
SP_LABELS = ["[2,4)", "[4,8)", "[8,16)"]


def assign_bins(values: np.ndarray, edges: list) -> np.ndarray:
    """Assign each value to a bin index based on edges (right edge exclusive)."""
    idx = np.zeros(len(values), dtype=int)
    for i in range(1, len(edges) - 1):
        idx[values >= edges[i]] = i
    return idx


class IndexedDataset(Dataset):
    """Wraps ModrecDataset, additionally yielding the global sample index."""

    def __init__(self, inner: ModrecDataset) -> None:
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, item: int) -> tuple:
        batch = self.inner[item]
        features, label = batch[0], batch[1]
        global_idx = int(self.inner.indices[item])
        return features, label, global_idx


def make_indexed_loader(
    signals: np.ndarray,
    metadata: pl.DataFrame,
    indices: np.ndarray,
    label_to_id: dict,
    model_name: str,
    channel_format: str,
    batch_size: int = 256,
    shuffle: bool = False,
) -> DataLoader:
    inner = ModrecDataset(
        signals=signals,
        metadata=metadata,
        indices=indices,
        label_to_id=label_to_id,
        representation=representation_for_model(model_name),
        channel_format=channel_format,
        remove_cfo=False,
        cfo_estimator="lag_correlation",
    )
    return DataLoader(
        IndexedDataset(inner),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def quick_train(
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
    snapshot_fn=None,   # called as snapshot_fn(epoch) after each epoch if not None
    snapshot_every: int = 1,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = n_total = 0
        bar = tqdm(loader, desc=f"train {epoch}/{epochs}", unit="batch", leave=False)
        for xb, yb, _ in bar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
            n_total += len(yb)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()
        print(f"  epoch {epoch:2d}/{epochs}  loss={total_loss / n_total:.4f}", end="")
        if snapshot_fn is not None and epoch % snapshot_every == 0:
            snapshot_fn(epoch)
            model.train()
        else:
            print()


def compute_losses(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (per_sample_loss, global_indices) arrays."""
    model.eval()
    losses, indices = [], []
    with torch.no_grad():
        for xb, yb, idxs in tqdm(loader, desc="eval", unit="batch", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            batch_losses = F.cross_entropy(model(xb), yb, reduction="none")
            losses.extend(batch_losses.cpu().tolist())
            indices.extend(idxs.tolist())
    return np.array(losses, dtype=np.float32), np.array(indices, dtype=np.int64)


def col_gradient_weights(
    losses: np.ndarray,
    global_indices: np.ndarray,
    metadata: pl.DataFrame,
    col_meta: str,
    col_edges: list,
    n_col: int,
) -> np.ndarray:
    """Return per-column gradient weight percentages (shape: n_col)."""
    col_vals = metadata[col_meta].to_numpy()[global_indices]
    col_bins = assign_bins(col_vals, col_edges)
    weights = np.array([
        losses[col_bins == j].sum() for j in range(n_col)
    ], dtype=np.float64)
    total = weights.sum()
    return (weights / total * 100.0) if total > 0 else weights


def compute_exact_grad_norms(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_subset: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-sample gradient L2 norms for a random subset (slow)."""
    all_x, all_y, all_idx = [], [], []
    for xb, yb, idxs in loader:
        all_x.append(xb)
        all_y.append(yb)
        all_idx.extend(idxs.tolist())
    all_x = torch.cat(all_x)
    all_y = torch.cat(all_y)
    all_idx = np.array(all_idx, dtype=np.int64)

    rng = np.random.default_rng(seed)
    subset = rng.choice(len(all_x), size=min(n_subset, len(all_x)), replace=False)

    model.train()
    grad_norms, sel_indices = [], []
    for i in tqdm(subset, desc="grad norms", unit="sample", leave=False):
        xb = all_x[i : i + 1].to(device)
        yb = all_y[i : i + 1].to(device)
        model.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5
        grad_norms.append(norm)
        sel_indices.append(int(all_idx[i]))

    return np.array(grad_norms, dtype=np.float32), np.array(sel_indices, dtype=np.int64)


def build_bin_grid(
    values: np.ndarray,
    global_indices: np.ndarray,
    metadata: pl.DataFrame,
    snr_bin_width: float,
    col_meta: str,
    col_edges: list,
    col_labels: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (grid, snr_labels, row_totals, col_totals, count_grid).
    grid shape: (n_snr_bins, n_col_bins), NaN where count == 0.
    """
    snr = metadata["snr_db"].to_numpy()[global_indices]
    col_vals = metadata[col_meta].to_numpy()[global_indices]

    snr_bin_starts = np.floor(snr / snr_bin_width) * snr_bin_width
    unique_snr_bins = sorted(np.unique(snr_bin_starts))
    snr_labels = [f"{int(s):+d}dB" for s in unique_snr_bins]
    col_bins = assign_bins(col_vals, col_edges)
    n_col = len(col_labels)

    n_snr = len(unique_snr_bins)
    grid_sum = np.zeros((n_snr, n_col))
    grid_cnt = np.zeros((n_snr, n_col), dtype=int)

    for i, snr_start in enumerate(unique_snr_bins):
        snr_mask = snr_bin_starts == snr_start
        for j in range(n_col):
            mask = snr_mask & (col_bins == j)
            if mask.sum() > 0:
                grid_sum[i, j] = values[mask].sum()
                grid_cnt[i, j] = mask.sum()

    grid_mean = np.where(grid_cnt > 0, grid_sum / grid_cnt, np.nan)
    row_totals = np.where(
        grid_cnt.sum(axis=1) > 0,
        np.nansum(grid_sum, axis=1) / np.nansum(grid_cnt, axis=1),
        np.nan,
    )
    col_totals = np.where(
        grid_cnt.sum(axis=0) > 0,
        np.nansum(grid_sum, axis=0) / np.nansum(grid_cnt, axis=0),
        np.nan,
    )
    return grid_mean, snr_labels, row_totals, col_totals, grid_cnt


def print_table(
    grid: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    fmt: str,
    row_totals: np.ndarray | None = None,
    col_totals: np.ndarray | None = None,
    suffix: str = "",
    row_axis_label: str = "SNR",
    col_axis_label: str = "",
) -> None:
    col_w = 9
    row_w = 9
    has_totals = row_totals is not None
    n_cols = len(col_labels) + (1 if has_totals else 0)
    width = row_w + col_w * n_cols

    print(f"\n{title}")
    print("=" * width)
    corner = f"{row_axis_label}" + (f"\\{col_axis_label}" if col_axis_label else "")
    header = f"{corner:>{row_w}}" + "".join(f"{c:>{col_w}}" for c in col_labels)
    if has_totals:
        header += f"{'ALL':>{col_w}}"
    print(header)
    print("-" * width)

    for i, rl in enumerate(row_labels):
        cells = []
        for j in range(len(col_labels)):
            v = grid[i, j]
            cells.append(f"{v:{col_w}{fmt}}{suffix}" if not np.isnan(v) else f"{'---':>{col_w}}")
        row = f"{rl:>{row_w}}" + "".join(cells)
        if has_totals:
            v = row_totals[i]
            row += f"{v:{col_w}{fmt}}{suffix}" if not np.isnan(v) else f"{'---':>{col_w}}"
        print(row)

    if col_totals is not None:
        print("-" * width)
        cells = [f"{col_totals[j]:{col_w}{fmt}}{suffix}" for j in range(len(col_labels))]
        print(f"{'ALL':>{row_w}}" + "".join(cells))

    print("=" * width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model", default="resnet_1d")
    parser.add_argument("--channel-format", default=None, help="Override channel format")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (skipped if --checkpoint given)")
    parser.add_argument("--checkpoint", default=None, help="Load model state_dict instead of training")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--snr-bin-width", type=float, default=SNR_BIN_WIDTH)
    parser.add_argument("--col-axis", default="symbol_period", choices=["symbol_period", "osr"],
                        help="Column axis for breakdown table (default: symbol_period)")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Fraction held out for analysis (not trained on)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-every", type=int, default=2,
                        help="Record gradient weight trajectory every N epochs (0 = only before/after)")
    parser.add_argument("--exact-grads", action="store_true", help="Also compute exact per-sample gradient norms")
    parser.add_argument("--grad-subset", type=int, default=500, help="Samples for exact gradient computation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    signals, metadata = load_dataset(args.dataset_dir)
    labels = metadata["modulation"].to_list()
    unique_labels = sorted(set(labels))
    ordered = [l for l in ("2PSK", "4PSK", "8PSK", "pi/4-DQPSK", "16QAM", "64QAM", "256QAM", "MSK") if l in unique_labels]
    ordered += [l for l in unique_labels if l not in ordered]
    label_to_id = {l: i for i, l in enumerate(ordered)}
    n_classes = len(label_to_id)

    all_idx = np.arange(len(metadata))
    train_idx, val_idx = train_test_split(all_idx, test_size=args.val_frac, random_state=args.seed)

    model_name = args.model
    channel_format = args.channel_format or required_channel_format_for(model_name) or "real_imag"
    representation = representation_for_model(model_name)
    in_channels = input_channels_for(representation, channel_format)
    n_samples = signals.shape[1]

    model, _ = make_model(model_name, n_classes=n_classes, n_samples=n_samples, in_channels=in_channels)
    model.to(device)

    train_loader = make_indexed_loader(
        signals, metadata, train_idx, label_to_id, model_name, channel_format,
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = make_indexed_loader(
        signals, metadata, val_idx, label_to_id, model_name, channel_format,
        batch_size=args.batch_size, shuffle=False,
    )

    if args.col_axis == "symbol_period":
        col_edges, col_labels, col_meta = SP_EDGES, SP_LABELS, "symbol_period"
        col_axis_label = "symbol_period"
    else:
        col_edges, col_labels, col_meta = OSR_EDGES, OSR_LABELS, "osr"
        col_axis_label = "osr"

    n_col = len(col_labels)
    trajectory: list[tuple[int, np.ndarray]] = []   # (epoch, col_weights)

    def snapshot(epoch: int) -> None:
        l, gi = compute_losses(model, val_loader, device)
        w = col_gradient_weights(l, gi, metadata, col_meta, col_edges, n_col)
        trajectory.append((epoch, w))
        bar = "  ".join(f"{col_labels[j]} {w[j]:4.1f}%" for j in range(n_col))
        print(f"  ← {bar}")

    # epoch-0 snapshot (random init)
    snapshot(0)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
        snapshot(args.epochs)
    else:
        print(f"\nTraining {model_name} for {args.epochs} epochs on {len(train_idx)} samples …")
        snap_every = args.snapshot_every if args.snapshot_every > 0 else args.epochs + 1
        quick_train(model, train_loader, args.epochs, device,
                    snapshot_fn=snapshot, snapshot_every=snap_every)
        # ensure final epoch is captured if not already
        if not trajectory or trajectory[-1][0] != args.epochs:
            snapshot(args.epochs)

    print(f"\nComputing per-sample losses on val set ({len(val_idx)} samples) …")
    losses, global_idx = compute_losses(model, val_loader, device)

    print(f"\nOverall mean NLL:  {losses.mean():.4f} nats  ({losses.mean() / np.log(2):.4f} bits)")
    print(f"Chance-level NLL:  {np.log(n_classes):.4f} nats  ({np.log2(n_classes):.4f} bits)")

    # Trajectory table
    col_w = 8
    row_w = 7
    hdr = f"{'epoch':>{row_w}}" + "".join(f"{c:>{col_w}}" for c in col_labels)
    width = row_w + col_w * n_col
    print(f"\nGRADIENT WEIGHT TRAJECTORY  [col={col_axis_label}]")
    print("=" * width)
    print(hdr)
    print("-" * width)
    for ep, w in trajectory:
        tag = " ← random init" if ep == 0 else ""
        print(f"{ep:>{row_w}d}" + "".join(f"{w[j]:>{col_w}.1f}%" for j in range(n_col)) + tag)
    print("=" * width)

    grid, snr_labels, row_totals, col_totals, count_grid = build_bin_grid(
        losses, global_idx, metadata, args.snr_bin_width,
        col_meta, col_edges, col_labels,
    )

    shared = dict(row_totals=row_totals, col_totals=col_totals,
                  row_axis_label="SNR", col_axis_label=col_axis_label)

    print_table(
        grid, snr_labels, col_labels,
        title=f"MEAN NLL (nats) — proxy for per-sample gradient magnitude  [col={col_axis_label}]",
        fmt=".3f", **shared,
    )

    # Gradient weight: mean_loss × count / total_loss_mass
    total_mass = np.nansum(grid * count_grid)
    weight_pct = np.where(count_grid > 0, grid * count_grid / total_mass * 100.0, np.nan)
    row_w_totals = np.array([np.nansum(weight_pct[i]) for i in range(len(snr_labels))])
    col_w_totals = np.array([np.nansum(weight_pct[:, j]) for j in range(n_col)])
    print_table(
        weight_pct, snr_labels, col_labels,
        title=f"GRADIENT WEIGHT (%) = mean_loss × count / total_loss_mass  [col={col_axis_label}]",
        fmt=".1f",
        row_totals=row_w_totals, col_totals=col_w_totals,
        row_axis_label="SNR", col_axis_label=col_axis_label,
    )

    print_table(
        count_grid.astype(float), snr_labels, col_labels,
        title=f"SAMPLE COUNT per bin  [col={col_axis_label}]",
        fmt=".0f",
        row_totals=count_grid.sum(axis=1).astype(float),
        col_totals=count_grid.sum(axis=0).astype(float),
        row_axis_label="SNR", col_axis_label=col_axis_label,
    )

    # Summary line: per-column gradient weight totals to highlight imbalance
    print(f"\nColumn gradient weight summary ({col_axis_label}):")
    for label, w in zip(col_labels, col_w_totals):
        bar = "#" * int(round(w / 2))
        print(f"  {label:>8s}  {w:5.1f}%  {bar}")

    if args.exact_grads:
        print(f"\nComputing exact gradient norms for {args.grad_subset} random val samples …")
        grad_norms, grad_idx = compute_exact_grad_norms(
            model, val_loader, device, args.grad_subset, args.seed
        )
        g_grid, _, g_row_totals, g_col_totals, _ = build_bin_grid(
            grad_norms, grad_idx, metadata, args.snr_bin_width,
            col_meta, col_edges, col_labels,
        )
        print_table(
            g_grid, snr_labels, col_labels,
            title=f"MEAN GRADIENT L2 NORM (exact, random subset)  [col={col_axis_label}]",
            fmt=".4f",
            row_totals=g_row_totals, col_totals=g_col_totals,
            row_axis_label="SNR", col_axis_label=col_axis_label,
        )


if __name__ == "__main__":
    main()
