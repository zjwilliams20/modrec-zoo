import argparse
from pathlib import Path

from modreczoo.data import load_dataset
from modreczoo.plotting import plot_example_spectrograms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write example spectrogram images to disk.")
    parser.add_argument("--dataset-dir", default="data/spectrogram_high_snr_sobol_16384")
    parser.add_argument("--output-dir", default="plots/spectrograms")
    parser.add_argument("--n-per-class", type=int, default=3, help="Examples per modulation (spread across SNR range).")
    parser.add_argument("--nperseg", type=int, default=64)
    parser.add_argument("--noverlap", type=int, default=48)
    parser.add_argument("--freq-bins", type=int, default=64)
    parser.add_argument("--time-bins", type=int, default=64)
    parser.add_argument("--window", default="kaiser")
    parser.add_argument("--window-beta", type=float, default=15.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    signals, metadata = load_dataset(args.dataset_dir)
    output_dir = Path(args.output_dir)
    plot_example_spectrograms(
        signals=signals,
        metadata=metadata,
        output_dir=output_dir,
        n_per_class=args.n_per_class,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        freq_bins=args.freq_bins,
        time_bins=args.time_bins,
        window=args.window,
        window_beta=args.window_beta,
    )
    files = sorted(output_dir.glob("*.png"))
    print(f"Wrote {len(files)} files to {output_dir}/")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
