#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-data/spectrogram_high_snr_sobol_16384}"
MAX_EXAMPLES="${MAX_EXAMPLES:-10000}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-11}"
CHANNEL_FORMAT="${CHANNEL_FORMAT:-mag_phase}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_ARGS=(
  --dataset-dir "$DATASET_DIR"
  --models spectrogram_cnn
  --channel-format "$CHANNEL_FORMAT"
  --cfo-estimator raw
  --max-examples "$MAX_EXAMPLES"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --seed "$SEED"
  --spectrogram-window kaiser
  --spectrogram-window-beta 15
)

CONFIGS=(
  "square_control 64  64  64 48"
  "freq96_t64    96  64  64 48"
  "freq128_t64   128 64  64 48"
  "freq96_t48    96  48  64 48"
  "freq128_t48   128 48  64 48"
  "time96        64  96  32 24"
  "time128       64  128 32 24"
)

printf "%-15s %4s %4s %8s %8s\n" name freq time nperseg noverlap
printf "%-15s %4s %4s %8s %8s\n" ---- ---- ---- ------- --------

for config in "${CONFIGS[@]}"; do
  read -r name freq_bins time_bins nperseg noverlap <<< "$config"
  printf "%-15s %4s %4s %8s %8s\n" "$name" "$freq_bins" "$time_bins" "$nperseg" "$noverlap"

  cmd=(
    uv run modreczoo-train
    "${COMMON_ARGS[@]}"
    --spectrogram-freq-bins "$freq_bins"
    --spectrogram-time-bins "$time_bins"
    --spectrogram-nperseg "$nperseg"
    --spectrogram-noverlap "$noverlap"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "DRY_RUN:"
    printf " %q" "${cmd[@]}"
    printf "\n"
  else
    "${cmd[@]}"
  fi
done
