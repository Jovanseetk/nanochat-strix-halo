#!/usr/bin/env bash
# Small single-GPU pretraining run. Install first with setup_strix_halo.sh.
# Override training options by appending base_train arguments, e.g.:
# bash runs/strix_halo.sh --num-iterations=1000 --device-batch-size=8
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -x .venv/bin/python ]]; then
    echo "Run bash runs/setup_strix_halo.sh first." >&2
    exit 1
fi
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat-strix-halo}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Fail before downloading data if the ROCm build or GPU access is broken.
python -m scripts.smoke_test --device-type cuda --require-rocm
python -m nanochat.dataset -n "${NANOCHAT_DATA_SHARDS:-2}"
if [[ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" || ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]]; then
    python -m scripts.tok_train --max-chars=100000000
fi

python -m scripts.base_train \
    --device-type=cuda \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=4 \
    --total-batch-size=8192 \
    --num-iterations=100 \
    --eval-every=50 \
    --eval-tokens=8192 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --model-tag=strix-halo \
    --run="${WANDB_RUN:-dummy}" \
    "$@"

echo "Base checkpoint saved under $NANOCHAT_BASE_DIR/base_checkpoints/strix-halo (unless --model-tag was overridden)."
echo "See docs/strix-halo.md for evaluation, SFT, and chat commands."
