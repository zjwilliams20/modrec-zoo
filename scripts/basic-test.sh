#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-data/basic_test_2048}"

uv run modreczoo-simulate generate --output-dir "$DATASET_DIR" --snr-range 0,30 --n-signals 200000 --n-samples 2048 --sampler sobol

uv run modreczoo-train --command sweep --dataset-dir "$DATASET_DIR" --device cuda --sample-frac 0.25 --batch-size 512 --seed 123 --sweep-channel-formats real_imag mag_phase --epochs 10 --models time_cnn frequency_cnn spectrogram_cnn --sweep-cfo-estimators raw
