#!/usr/bin/env bash
set -euo pipefail

# Basic parameter search for spectrogram_resnet.
#
# Sweeps the three main axes independently (kernel shape, model capacity,
# channel format) with a shared fixed preprocessing baseline. Each group
# can be skipped via the RUN_GROUPS env var (see below).
#
# Env vars:
#   DATASET_DIR   – path to dataset                  (default: data/spectrogram_high_snr_sobol_16384)
#   MAX_EXAMPLES  – cap on training examples          (default: 10000)
#   EPOCHS        – epochs per run                    (default: 12)
#   BATCH_SIZE                                        (default: 512)
#   NUM_WORKERS                                       (default: 4)
#   DEVICE                                            (default: cuda)
#   SEED                                              (default: 11)
#   RUN_GROUPS        – space-separated subset of groups to run: kernels capacity formats
#                   (default: all three)
#   DRY_RUN       – print commands instead of running (default: 0)

DATASET_DIR="${DATASET_DIR:-data/spectrogram_high_snr_sobol_16384}"
MAX_EXAMPLES="${MAX_EXAMPLES:-16384}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-11}"
RUN_GROUPS="${RUN_GROUPS:-kernels capacity formats}"
DRY_RUN="${DRY_RUN:-0}"

# Fixed preprocessing baseline (kaiser15 was best in prior window sweep)
COMMON_ARGS=(
  --dataset-dir "$DATASET_DIR"
  --models spectrogram_resnet
  --cfo-estimator raw
  --max-examples "$MAX_EXAMPLES"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --seed "$SEED"
  --spectrogram-size 64
  --spectrogram-nperseg 64
  --spectrogram-noverlap 48
  --spectrogram-window kaiser
  --spectrogram-window-beta 15
  --spectrogram-base-channels 32
  --channel-format mag_phase
)

run_cmd() {
  local cmd=("${@}")
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "DRY_RUN:"
    printf " %q" "${cmd[@]}"
    printf "\n"
  else
    "${cmd[@]}"
  fi
}

# ── Group 1: kernel shape ────────────────────────────────────────────────────
# Fixed: base_channels=32, channel_format=mag_phase
# Explores symmetric vs increasingly asymmetric (freq_kernel >= time_kernel)
run_kernels() {
  echo
  echo "=== Group: kernels (base_channels=32, mag_phase) ==="
  printf "%-18s %10s %10s\n" name freq_kernel time_kernel
  printf "%-18s %10s %10s\n" ---- ----------- -----------

  # name   fk  tk
  local CONFIGS=(
    "sym_3x3       3 3"
    "asym_5x3      5 3"
    "asym_7x3      7 3"
    "asym_7x5      7 5"
    "asym_9x3      9 3"
  )

  for config in "${CONFIGS[@]}"; do
    read -r name fk tk <<< "$config"
    printf "%-18s %10s %10s\n" "$name" "$fk" "$tk"
    run_cmd uv run modreczoo-train \
      "${COMMON_ARGS[@]}" \
      --spectrogram-freq-kernel "$fk" \
      --spectrogram-time-kernel "$tk" \
      --run-name "kernels-${name}"
  done
}

# ── Group 2: model capacity ──────────────────────────────────────────────────
# Fixed: freq_kernel=5, time_kernel=3, channel_format=mag_phase
run_capacity() {
  echo
  echo "=== Group: capacity (fk=5 tk=3, mag_phase) ==="
  printf "%-18s %13s %16s\n" name base_channels blocks_per_stage
  printf "%-18s %13s %16s\n" ---- ------------- ----------------

  # name   base_channels  b0 b1 b2 b3
  local CONFIGS=(
    "c24_shallow    24 1 1 1 1"
    "c24_default    24 2 2 2 2"
    "c32_shallow    32 1 1 1 1"
    "c32_default    32 2 2 2 2"
    "c32_deep       32 2 2 3 3"
    "c48_default    48 2 2 2 2"
  )

  for config in "${CONFIGS[@]}"; do
    read -r name bc b0 b1 b2 b3 <<< "$config"
    printf "%-18s %13s %16s\n" "$name" "$bc" "$b0 $b1 $b2 $b3"
    run_cmd uv run modreczoo-train \
      "${COMMON_ARGS[@]}" \
      --spectrogram-freq-kernel 5 \
      --spectrogram-time-kernel 3 \
      --spectrogram-base-channels "$bc" \
      --spectrogram-blocks-per-stage "$b0" "$b1" "$b2" "$b3" \
      --run-name "capacity-${name}"
  done
}

# ── Group 3: channel format ──────────────────────────────────────────────────
# Fixed: freq_kernel=5, time_kernel=3, base_channels=32
run_formats() {
  echo
  echo "=== Group: formats (fk=5 tk=3, base_channels=32) ==="
  printf "%-18s %13s\n" name channel_format
  printf "%-18s %13s\n" ---- --------------

  local CONFIGS=(
    "mag             mag"
    "mag_phase       mag_phase"
    "mag_inst_freq   mag_inst_freq"
    "real_imag       real_imag"
  )

  for config in "${CONFIGS[@]}"; do
    read -r name fmt <<< "$config"
    printf "%-18s %13s\n" "$name" "$fmt"
    run_cmd uv run modreczoo-train \
      "${COMMON_ARGS[@]}" \
      --spectrogram-freq-kernel 5 \
      --spectrogram-time-kernel 3 \
      --channel-format "$fmt" \
      --run-name "formats-${name}"
  done
}

for group in $RUN_GROUPS; do
  case "$group" in
    kernels)  run_kernels  ;;
    capacity) run_capacity ;;
    formats)  run_formats  ;;
    *) echo "Unknown group '$group'. Valid: kernels capacity formats" >&2; exit 1 ;;
  esac
done
