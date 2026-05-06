import argparse

import numpy as np
import torch

from modreczoo.data import load_dataset, ordered_modulation_labels
from modreczoo.training import (
    CFO_SWEEP_MODES,
    CHANNEL_FORMATS,
    MODEL_NAMES,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ModRec baselines.")
    parser.add_argument("command", nargs="?", choices=("train", "sweep"), default="train")
    parser.add_argument("--dataset-dir", default="data/awgn_snr0_30")
    parser.add_argument(
        "--test-dataset-dir",
        default=None,
        help="Optional external dataset for final test/OOD evaluation. Defaults to a held-out split of --dataset-dir.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["time_cnn", "resnet_1d", "dilated_cnn_1d"],
        choices=MODEL_NAMES,
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
    parser.add_argument("--channel-format", choices=CHANNEL_FORMATS, default="real_imag")
    parser.add_argument("--remove-cfo", action="store_true")
    parser.add_argument(
        "--cfo-estimator",
        choices=CFO_SWEEP_MODES,
        default="lag_correlation",
    )
    parser.add_argument("--sweep-channel-formats", nargs="+", choices=CHANNEL_FORMATS, default=list(CHANNEL_FORMATS))
    parser.add_argument("--sweep-cfo-estimators", nargs="+", choices=CFO_SWEEP_MODES, default=list(CFO_SWEEP_MODES))
    parser.add_argument("--sweep-batch-sizes", nargs="+", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_signals, train_metadata = load_dataset(args.dataset_dir)
    observed_labels = train_metadata["modulation"].unique().to_list()
    labels = ordered_modulation_labels(observed_labels)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

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

    configure_mlflow()

    configs = iter_sweep_args(args) if args.command == "sweep" else [args]
    print(f"Running {len(configs)} configuration(s).")
    for sweep_index, cfg in enumerate(configs, start=1):
        for model_name in cfg.models:
            print(
                f"[{sweep_index}/{len(configs)}] {run_name_for(cfg, model_name)} "
                f"epochs={cfg.epochs} lr={cfg.lr:g} seed={cfg.seed}"
            )
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
            )


if __name__ == "__main__":
    main()
