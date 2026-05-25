#!/bin/bash

n_workers=${1:-4}
uv run modreczoo-simulate generate --output-dir data/baseline_4096 --snr-range 0,30 --n-signals 200000 --n-samples 4096 --sampler random --channel awgn --num-workers $n_workers
uv run modreczoo-simulate generate --output-dir data/baseline_4096_channels --snr-range 0,30 --n-signals 200000 --n-samples 4096 --sampler random --channel awgn rayleigh rician soft_limiter --num-workers $n_workers
uv run modreczoo-simulate generate --output-dir data/baseline_32768 --snr-range 0,30 --n-signals 40000 --n-samples 32768 --sampler random --channel awgn --num-workers $n_workers
uv run scripts/convert_cspb_2018r2.py --input-dir data/cspb_2018r2/delivered/ --output-dir data/cspb_2018r2/ --force --batch-fraction 0.2
