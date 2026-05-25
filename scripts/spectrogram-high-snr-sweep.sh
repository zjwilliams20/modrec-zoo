#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-data/baseline_4096}"
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
)

CONFIGS=(
  "baseline        64  64  48 hann   0"
  "overlap50       64  64  32 hann   0"
  "short_dense     64  32  24 hann   0"
  "short_overlap50 64  32  16 hann   0"
  "wide_dense      128 128 96 hann   0"
  "wide_overlap50  128 128 64 hann   0"
  "kaiser5         64  64  48 kaiser 5"
  "kaiser10        64  64  48 kaiser 10"
  "kaiser15        64  64  48 kaiser 15"
)

printf "%-16s %4s %8s %8s %-10s\n" name size nperseg noverlap window
printf "%-16s %4s %8s %8s %-10s\n" ---- ---- ------- -------- ------

for config in "${CONFIGS[@]}"; do
  read -r name size nperseg noverlap window beta <<< "$config"
  if [[ "$window" == "kaiser" ]]; then
    window="kaiser:${beta}"
  fi
  printf "%-16s %4s %8s %8s %-10s\n" "$name" "$size" "$nperseg" "$noverlap" "$window"

  cmd=(
    uv run modreczoo-train
    "${COMMON_ARGS[@]}"
    --spectrogram-freq-bins "$size"
    --spectrogram-time-bins "$size"
    --spectrogram-nperseg "$nperseg"
    --spectrogram-noverlap "$noverlap"
    --spectrogram-window "$window"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "DRY_RUN:"
    printf " %q" "${cmd[@]}"
    printf "\n"
  else
    "${cmd[@]}"
  fi
done
