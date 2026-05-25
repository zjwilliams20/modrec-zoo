import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from modreczoo.data import EXTRAS_FILE, METADATA_FILE, SIGNALS_FILE, save_dataset
from modreczoo.simulation import (
    DEFAULT_PARAMS,
    MODULATIONS,
    SUPPORTED_MODULATIONS,
    ber_sweep,
    generate_dataset,
)


def parse_range(value: str, cast=float) -> Tuple:
    parts = [cast(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Ranges must be provided as low,high.")
    return tuple(parts)


def parse_modulations(value: str) -> Tuple[str, ...]:
    mods = tuple("pi/4-DQPSK" if part.strip() == "DQPSK" else part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(mods) - set(SUPPORTED_MODULATIONS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unsupported modulation(s): {', '.join(unknown)}")
    return mods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic RF modulation-recognition simulator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate an RFML dataset.")
    generate.add_argument("--output-dir", default="dataset")
    generate.add_argument("--n-signals", type=int, default=1000)
    generate.add_argument("--n-samples", type=int, default=DEFAULT_PARAMS["n_samples"])
    generate.add_argument("--modulations", type=parse_modulations, default=",".join(MODULATIONS))
    generate.add_argument("--snr-range", type=lambda v: parse_range(v, float), default=DEFAULT_PARAMS["snr_range"])
    generate.add_argument("--cfo-range", type=lambda v: parse_range(v, float), default=DEFAULT_PARAMS["cfo_range"])
    generate.add_argument("--cpo-range", type=lambda v: parse_range(v, float), default=DEFAULT_PARAMS["cpo_range"])
    generate.add_argument("--sto-range", type=lambda v: parse_range(v, float), default=DEFAULT_PARAMS["sto_range"])
    generate.add_argument(
        "--symbol-period-range",
        type=lambda v: parse_range(v, int),
        default=DEFAULT_PARAMS["symbol_period_range"],
    )
    generate.add_argument("--osr-range", type=lambda v: parse_range(v, float), default=DEFAULT_PARAMS["osr_range"])
    generate.add_argument("--ebw-range", type=lambda v: parse_range(v, float), default=DEFAULT_PARAMS["ebw_range"])
    generate.add_argument(
        "--channel",
        choices=("awgn", "rayleigh", "rician", "soft_limiter"),
        nargs="+",
        default=[DEFAULT_PARAMS["channel"]],
    )
    generate.add_argument("--sampler", choices=("sobol", "random"), default=DEFAULT_PARAMS["sampler"])
    generate.add_argument("--num-workers", type=int, default=1)
    generate.add_argument("--rician-k-range", type=lambda v: parse_range(v, float), default=DEFAULT_PARAMS["rician_k_range"])
    generate.add_argument("--n-taps-range", type=lambda v: parse_range(v, int), default=DEFAULT_PARAMS["n_taps_range"])
    generate.add_argument(
        "--delay-spread-symbols-range",
        type=lambda v: parse_range(v, float),
        default=DEFAULT_PARAMS["delay_spread_symbols_range"],
    )
    generate.add_argument(
        "--delay-decay-symbols-range",
        type=lambda v: parse_range(v, float),
        default=DEFAULT_PARAMS["delay_decay_symbols_range"],
    )
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--debug", action="store_true")

    ber = subparsers.add_parser("ber", help="Print empirical BER against a simple AWGN theoretical model.")
    ber.add_argument("--modulations", type=parse_modulations, default="2PSK,4PSK,8PSK,pi/4-DQPSK,16QAM,64QAM")
    ber.add_argument("--ebn0-db", type=float, nargs="+", default=[0, 4, 8, 12, 16])
    ber.add_argument("--n-bits", type=int, default=200_000)
    ber.add_argument("--seed", type=int, default=0)
    ber.add_argument("--output-csv", default=None)

    plot = subparsers.add_parser("plot", help="Save waveform summary plots by modulation.")
    plot.add_argument("--output-dir", default="plots")
    plot.add_argument("--modulations", type=parse_modulations, default=",".join(MODULATIONS))
    plot.add_argument("--k-symbols", type=int, default=128)
    plot.add_argument("--osr", type=int, default=8)
    plot.add_argument("--ebw", type=float, default=0.35)
    plot.add_argument("--ebn0-db", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    plot.add_argument("--ber-bits", type=int, default=100_000)
    plot.add_argument("--seed", type=int, default=0)
    plot.add_argument("--show", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "generate":
        params = dict(DEFAULT_PARAMS)
        params.update(
            {
                "n_samples": args.n_samples,
                "snr_range": args.snr_range,
                "cfo_range": args.cfo_range,
                "cpo_range": args.cpo_range,
                "sto_range": args.sto_range,
                "symbol_period_range": args.symbol_period_range,
                "osr_range": args.osr_range,
                "ebw_range": args.ebw_range,
                "channel": args.channel,
                "sampler": args.sampler,
                "rician_k_range": args.rician_k_range,
                "n_taps_range": args.n_taps_range,
                "delay_spread_symbols_range": args.delay_spread_symbols_range,
                "delay_decay_symbols_range": args.delay_decay_symbols_range,
                "seed": args.seed,
                "num_workers": args.num_workers,
            }
        )
        signals, metadata, extras = generate_dataset(
            args.modulations,
            args.n_signals,
            params,
            debug=args.debug,
            num_workers=args.num_workers,
        )
        save_dataset(args.output_dir, signals, metadata, extras=extras)

        def _git(*cmd: str) -> str:
            try:
                return subprocess.check_output(["git", *cmd], text=True, stderr=subprocess.DEVNULL).strip()
            except Exception:
                return "unknown"

        manifest = {
            "signals": SIGNALS_FILE,
            "metadata": METADATA_FILE,
            "params": params,
            "modulations": list(args.modulations),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_hash": _git("rev-parse", "HEAD"),
            "git_dirty": _git("status", "--porcelain") != "",
        }
        if extras:
            manifest["extras"] = EXTRAS_FILE
        with open(Path(args.output_dir) / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote {len(metadata)} signals to {os.path.abspath(args.output_dir)}")
        print(metadata.head())

    if args.command == "ber":
        results = ber_sweep(args.modulations, args.ebn0_db, args.n_bits, args.seed)
        print(results)
        if args.output_csv:
            results.write_csv(args.output_csv)

    if args.command == "plot":
        from modreczoo.plotting import plot_modulation_summaries

        plot_modulation_summaries(
            modulations=args.modulations,
            output_dir=args.output_dir,
            k_symbols=args.k_symbols,
            osr=args.osr,
            ebw=args.ebw,
            ebn0_db=args.ebn0_db,
            ber_bits=args.ber_bits,
            seed=args.seed,
            show=args.show,
        )
        print(f"Wrote plots to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
