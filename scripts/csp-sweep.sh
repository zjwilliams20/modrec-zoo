#!/usr/bin/env bash
# Sweep of CSP-inspired architectures against the best-performing baselines.
#
# New models (forced formats):
#   cpowers_resnet_1d  — complex_powers (Re/Im at orders 1, 2, 4)
#   multilag_net_1d    — real_imag (multi-lag conjugate products, model-internal)
#   cyclic_caf_1d      — real_imag (CAF magnitude spectra, model-internal)
#   scf_resnet         — scf (2D spectral correlation function image)
#
# Baselines for comparison (best from prior sweeps):
#   apf_net_1d         — apf (forced)
#   diff_resnet_1d     — differential_complex (forced)
#   resnet_1d          — differential_complex (best general-purpose format)
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-data/awgn_snr0_30}"
MAX_EXAMPLES="${MAX_EXAMPLES:-16384}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-modrec-csp}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
  uv run modreczoo-train
  --command sweep
  --dataset-dir "$DATASET_DIR"
  --models
    cpowers_resnet_1d
    multilag_net_1d
    cyclic_caf_1d
    scf_resnet
    apf_net_1d
    diff_resnet_1d
    resnet_1d
  --sweep-channel-formats differential_complex
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --seed "$SEED"
  --experiment-name "$EXPERIMENT_NAME"
  --max-examples "$MAX_EXAMPLES"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf "DRY_RUN:"
  printf " %q" "${cmd[@]}"
  printf "\n"
else
  "${cmd[@]}"
fi
