#!/usr/bin/env bash
# Sweep of CSP-inspired architectures against the best-performing baselines.
#
# Architecturally fused models (format is part of the model design):
#   apf_net_1d         — apf (forced: 4-channel stream encoder)
#   scf_resnet         — scf (forced: 2D spectral correlation image)
#
# Flexible baselines (swept over relevant formats):
#   resnet_1d          — differential_complex, complex_powers, multilag, cyclic_caf
#   multi_stream_1d    — differential_complex, complex_powers, multilag, cyclic_caf
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
    apf_net_1d
    resnet_1d
    multi_stream_1d
  --sweep-channel-formats differential_complex complex_powers multilag cyclic_caf
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
