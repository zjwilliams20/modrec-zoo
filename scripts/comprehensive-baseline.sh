#!/usr/bin/env bash
# Comprehensive baseline: all models × all channel formats.
#
# Fixed-format models (format forced by registry) run once each:
#   apf_net_1d   → apf
#   multilag_net_1d → multilag
#   cyclic_caf_1d   → cyclic_caf
#   scf_resnet      → scf
#
# Flexible models are swept over all channel formats.
#
# Usage:
#   DATASET_DIR=data/baseline_4096 bash scripts/comprehensive-baseline.sh
#   DATASET_DIR=data/foo EXTRA_TEST_DIRS="data/bar data/baz" bash scripts/comprehensive-baseline.sh
#   DRY_RUN=1 bash scripts/comprehensive-baseline.sh   # preview commands only
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-data/baseline_4096}"
EXTRA_TEST_DIRS="${EXTRA_TEST_DIRS:-data/baseline_4096_channels}"   # space-separated dataset dirs
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-comprehensive_baseline}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
DRY_RUN="${DRY_RUN:-0}"

FLEXIBLE_MODELS=(
  # time_cnn
  # frequency_cnn
  # spectrogram_cnn
  # spectrogram_resnet
  resnet_1d
  # complex_cnn_1d
  dilated_cnn_1d
  patch_transformer_1d
  multiscale_pyramid_1d
  multi_stream_1d
)

# All channel formats (fixed-format models simply override this internally).
ALL_FORMATS=(
  real_imag
  # mag_phase
  # differential_complex
  # apf
  complex_powers
  multilag
  # cyclic_caf
  # scf
)

# Fixed-format models — include in a single-format sweep so each runs once.
# Also includes feature_mlp: handcrafted_features ignores channel_format, so one run suffices.
FIXED_FORMAT_MODELS=(
  apf_net_1d
  multilag_net_1d
  cyclic_caf_1d
  scf_resnet
  iq_features_mlp
  csp_expert_mlp
)

# ── Shared args ────────────────────────────────────────────────────────────────
common_args=(
  --dataset-dir "$DATASET_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --seed "$SEED"
  --experiment-name "$EXPERIMENT_NAME"
)

[[ -n "$MAX_EXAMPLES" ]] && common_args+=(--max-examples "$MAX_EXAMPLES")

if [[ -n "$EXTRA_TEST_DIRS" ]]; then
  read -ra _extra <<< "$EXTRA_TEST_DIRS"
  common_args+=(--extra-test-dirs "${_extra[@]}")
fi

n_flexible=$(( ${#FLEXIBLE_MODELS[@]} * ${#ALL_FORMATS[@]} ))
n_fixed=${#FIXED_FORMAT_MODELS[@]}
echo "Planned runs: ${n_flexible} flexible (${#FLEXIBLE_MODELS[@]} models × ${#ALL_FORMATS[@]} formats) + ${n_fixed} fixed-format = $(( n_flexible + n_fixed )) total"
echo "Dataset: $DATASET_DIR | epochs=$EPOCHS batch=$BATCH_SIZE device=$DEVICE seed=$SEED"
[[ -n "$EXTRA_TEST_DIRS" ]] && echo "Extra test dirs: $EXTRA_TEST_DIRS"
echo

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "DRY_RUN:"
    printf " %q" "$@"
    printf "\n"
  else
    "$@"
  fi
}

echo "=== Pass 1/2: Flexible models × all formats ==="
run uv run modreczoo-train \
  --command sweep \
  "${common_args[@]}" \
  --models "${FLEXIBLE_MODELS[@]}" \
  --sweep-channel-formats "${ALL_FORMATS[@]}"

echo
echo "=== Pass 2/2: Fixed-format models (one run each) ==="
run uv run modreczoo-train \
  --command sweep \
  "${common_args[@]}" \
  --models "${FIXED_FORMAT_MODELS[@]}" \
  --sweep-channel-formats real_imag
