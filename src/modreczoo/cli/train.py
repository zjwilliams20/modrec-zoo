import time
from jsonargparse import ArgumentParser
from typing import List
import argparse

import numpy as np
import torch

from modreczoo.data import load_dataset, ordered_modulation_labels
from modreczoo.oracle import build_oracle_cache
from modreczoo.training import (
    CFO_SWEEP_MODES,
    CHANNEL_FORMATS,
    MODEL_NAMES,
    PREPROCESSOR_NAMES,
    configure_mlflow,
    dataset_sample_indices,
    iter_sweep_args,
    run_config,
    run_name_for,
    stratified_split,
    stratified_train_val_split,
    validate_args,
    validate_known_labels,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train ModRec baselines.")
    parser.add_argument("--config", action="config")
    parser.add_argument("--command", choices=("train", "sweep", "oracle-cache"), default="train")
    parser.add_argument("--experiment-name", default="modrec", help="MLflow experiment name.")
    parser.add_argument("--run-name", default=None, help="Override the auto-generated MLflow run name.")
    parser.add_argument("--dataset-dir", default="data/baseline_4096")
    parser.add_argument("--force-oracle-cache", action="store_true")
    parser.add_argument(
        "--test-dataset-dir",
        default=None,
        help="Optional external dataset for final test/OOD evaluation. Defaults to a held-out split of --dataset-dir.",
    )
    parser.add_argument(
        "--extra-test-dirs",
        nargs="+",
        default=None,
        help="Additional dataset directories evaluated after training, logged under their directory stem name.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["time_cnn", "resnet_1d", "dilated_cnn_1d"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channel-format", default="real_imag")
    parser.add_argument("--remove-cfo", type=bool, default=False)
    parser.add_argument("--cfo-estimator", default="lag_correlation")
    parser.add_argument("--spectrogram-freq-bins", type=int, default=64)
    parser.add_argument("--spectrogram-time-bins", type=int, default=64)
    parser.add_argument("--spectrogram-nperseg", type=int, default=64)
    parser.add_argument("--spectrogram-noverlap", type=int, default=48)
    parser.add_argument("--spectrogram-window", default="hann", help="Window name, or 'kaiser:14' to set kaiser beta.")
    parser.add_argument("--spectrogram-base-channels", type=int, default=24)
    parser.add_argument("--spectrogram-freq-kernel", type=int, default=7)
    parser.add_argument("--spectrogram-time-kernel", type=int, default=3)
    parser.add_argument("--transformer-patch-size", type=int, default=32    )
    parser.add_argument("--transformer-d-model", type=int, default=128)
    parser.add_argument("--transformer-n-heads", type=int, default=4)
    parser.add_argument("--transformer-n-layers", type=int, default=4)
    parser.add_argument(
        "--preprocessor",
        choices=PREPROCESSOR_NAMES,
        default="none",
        help="Optional differentiable Torch frontend. Defaults to the existing dataset preprocessing path.",
    )
    parser.add_argument("--preprocessor-channels", type=int, default=None)
    parser.add_argument("--preprocessor-kernel-size", type=int, default=31)
    parser.add_argument("--preprocessor-max-time-shift", type=float, default=8.0)
    parser.add_argument("--preprocessor-max-frequency-shift", type=float, default=0.02)
    parser.add_argument("--preprocessor-max-phase-shift", type=float, default=float(np.pi))
    parser.add_argument(
        "--aux-targets",
        nargs="+",
        default=None,
        help="Metadata columns to predict as auxiliary classification heads, e.g. snr_db osr ebw channel.",
    )
    parser.add_argument("--aux-bins", type=int, default=8, help="Quantile bins for numeric auxiliary metadata targets.")
    parser.add_argument("--aux-loss-weight", type=float, default=0.2)
    parser.add_argument("--aux-loss-mode", choices=("fixed", "uncertainty"), default="fixed")
    parser.add_argument("--aux-head-hidden", type=int, default=0)
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="SNR-driven focal loss exponent. 0 = standard cross-entropy. "
             "Higher values down-weight high-SNR (easy) samples more aggressively.",
    )
    parser.add_argument("--sweep-channel-formats", nargs="+", default=list(CHANNEL_FORMATS))
    parser.add_argument("--sweep-cfo-estimators", nargs="+", default=["lag_correlation"])
    parser.add_argument("--sweep-batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--sweep-spectrogram-freq-bins", nargs="+", type=int, default=None)
    parser.add_argument("--sweep-spectrogram-time-bins", nargs="+", type=int, default=None)
    parser.add_argument("--sweep-spectrogram-base-channels", nargs="+", type=int, default=None)
    parser.add_argument("--sweep-spectrogram-freq-kernels", nargs="+", type=int, default=None)
    parser.add_argument("--sweep-spectrogram-time-kernels", nargs="+", type=int, default=None)
    return parser


def _print_sweep_table(configs: List[argparse.Namespace]) -> None:
    from modreczoo.training import run_name_for

    _COLS = [
        ("run_name",               lambda c: run_name_for(c, c.models[0])),
        ("model",                  lambda c: c.models[0]),
        ("channel_format",         lambda c: c.channel_format),
        ("batch_size",             lambda c: str(c.batch_size)),
        ("cfo",                    lambda c: c.cfo_estimator if c.remove_cfo else "raw"),
        ("freq_bins",              lambda c: str(c.spectrogram_freq_bins)),
        ("time_bins",              lambda c: str(c.spectrogram_time_bins)),
        ("base_ch",                lambda c: str(c.spectrogram_base_channels)),
        ("freq_k",                 lambda c: str(c.spectrogram_freq_kernel)),
        ("time_k",                 lambda c: str(c.spectrogram_time_kernel)),
    ]

    rows = [[fn(c) for _, fn in _COLS] for c in configs]
    headers = [h for h, _ in _COLS]

    # Drop columns that are identical across all configs.
    keep = [i for i, col in enumerate(zip(*rows)) if len(set(col)) > 1]
    headers = [headers[i] for i in keep]
    rows = [[row[i] for i in keep] for row in rows]

    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print(sep)
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_yaml = parser.dump(args)
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_signals, train_metadata = load_dataset(args.dataset_dir)
    observed_labels = train_metadata["modulation"].unique().to_list()
    labels = ordered_modulation_labels(observed_labels)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    if args.command == "oracle-cache":
        result = build_oracle_cache(
            args.dataset_dir,
            train_signals,
            train_metadata,
            labels,
            force=args.force_oracle_cache,
            num_workers=args.num_workers,
        )
        print(
            f"Wrote oracle cache for {result['row_count']} rows: "
            f"{result['parquet_path']} and {result['json_path']} "
            f"(accuracy={result['accuracy']:.4f})."
        )
        return

    full_train_labels = train_metadata["modulation"].to_numpy()
    train_dataset_idx = dataset_sample_indices(full_train_labels, args.sample_frac, args.max_examples, args.seed)
    sampled_train_labels = full_train_labels[train_dataset_idx]

    args.val_dataset_dir_effective = args.dataset_dir
    if args.test_dataset_dir:
        test_signals, test_metadata = load_dataset(args.test_dataset_dir)
        validate_known_labels(test_metadata, labels, args.test_dataset_dir, "Test")
        full_test_labels = test_metadata["modulation"].to_numpy()
        test_idx = dataset_sample_indices(full_test_labels, args.sample_frac, args.max_examples, args.seed)
        train_rel_idx, val_rel_idx = stratified_train_val_split(
            sampled_train_labels,
            args.train_frac,
            args.val_frac,
            args.seed,
        )
        splits = (train_dataset_idx[train_rel_idx], train_dataset_idx[val_rel_idx], test_idx)
        args.test_dataset_dir_effective = args.test_dataset_dir
        args.test_dataset_source = "external_dataset"
    else:
        test_signals, test_metadata = train_signals, train_metadata
        relative_splits = stratified_split(sampled_train_labels, args.train_frac, args.val_frac, args.seed)
        splits = tuple(train_dataset_idx[idx] for idx in relative_splits)
        args.test_dataset_dir_effective = args.dataset_dir
        args.test_dataset_source = "heldout_split"

    val_signals, val_metadata = train_signals, train_metadata
    args.n_train_dataset_examples_available = int(train_signals.shape[0])
    args.n_train_dataset_examples_used = int(len(train_dataset_idx))
    args.n_test_dataset_examples_available = int(test_signals.shape[0])
    args.n_test_dataset_examples_used = int(len(splits[2]))
    args.n_train_examples = int(len(splits[0]))
    args.n_val_examples = int(len(splits[1]))
    args.n_test_examples = int(len(splits[2]))
    if args.test_dataset_dir:
        print(
            f"Using {args.n_train_dataset_examples_used}/{args.n_train_dataset_examples_available} training-source examples: "
            f"train={args.n_train_examples}, val={args.n_val_examples}."
        )
        print(
            f"Testing on external dataset {args.test_dataset_dir}: "
            f"{args.n_test_dataset_examples_used}/{args.n_test_dataset_examples_available} examples."
        )
    else:
        print(
            f"Using {args.n_train_dataset_examples_used}/{args.n_train_dataset_examples_available} examples: "
            f"train={args.n_train_examples}, val={args.n_val_examples}, test={args.n_test_examples}."
        )

    configure_mlflow(args.experiment_name)

    configs = iter_sweep_args(args) if args.command == "sweep" else [args]
    print(f"Running {len(configs)} configuration(s).")
    if args.command == "sweep":
        _print_sweep_table(configs)
    for sweep_index, cfg in enumerate(configs, start=1):
        for model_name in cfg.models:
            print(
                f"[{sweep_index}/{len(configs)}] {run_name_for(cfg, model_name)} "
                f"epochs={cfg.epochs} lr={cfg.lr:g} seed={cfg.seed}"
            )
            try:
                run_config(
                    cfg,
                    model_name,
                    train_signals,
                    train_metadata,
                    val_signals,
                    val_metadata,
                    test_signals,
                    test_metadata,
                    label_to_id,
                    id_to_label,
                    labels,
                    splits,
                    sweep_index,
                    len(configs),
                    config_yaml=config_yaml,
                    extra_test_dirs=args.extra_test_dirs,
                )
            except KeyboardInterrupt:
                print(f"\n  Skipping {run_name_for(cfg, model_name)}. Ctrl+C again within 2s to quit.")
                try:
                    time.sleep(2.0)
                except KeyboardInterrupt:
                    raise SystemExit(1)


if __name__ == "__main__":
    main()
