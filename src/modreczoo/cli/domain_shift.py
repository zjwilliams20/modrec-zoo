import argparse
from pathlib import Path

from modreczoo.domain_shift import parse_domain_specs, write_domain_shift_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an interactive latent-space domain-shift report.")
    parser.add_argument("--run-id", required=True, help="MLflow run id to analyze.")
    parser.add_argument(
        "--domains",
        nargs="+",
        required=True,
        help="Domains as name:auto for train/val/test reconstruction or name=dataset_dir.",
    )
    parser.add_argument("--source-domain", default=None, help="Reference domain for shift metrics. Defaults to first domain.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for report artifacts.")
    parser.add_argument("--max-examples-per-domain", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = write_domain_shift_report(
        run_id=args.run_id,
        domain_specs=parse_domain_specs(args.domains),
        output_dir=args.output_dir,
        source_domain=args.source_domain,
        max_examples_per_domain=args.max_examples_per_domain,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        seed=args.seed,
    )
    print("Wrote domain-shift artifacts:")
    for name, path in paths.items():
        print(f"  {name:14s} {path}")


if __name__ == "__main__":
    main()
