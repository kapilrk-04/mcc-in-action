#!/usr/bin/env bash
# Pre-download a model into the HF cache used by executor.py. Gated repos
# (e.g. meta-llama, "manual" approval) need HF_TOKEN to belong to an account
# whose access request has been approved.
#
#   bash download_model.sh                                        # Llama-3.1-70B-Instruct
#   MODEL=meta-llama/Llama-3.1-8B-Instruct bash download_model.sh
#
# Written to $HF_HUB_CACHE (default ../hf_models, the same default
# run_ablations.sh uses); Llama-3.1-70B is ~141 GB of safetensors. Needs GNU
# coreutils (Linux) for the df check.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO_ROOT/hf_models}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/hf_models}"

for envf in "$SCRIPT_DIR/.env" "$REPO_ROOT/.env"; do
  if [ -z "${HF_TOKEN:-}" ] && [ -f "$envf" ]; then
    # shellcheck disable=SC1090
    set -a; source "$envf"; set +a
  fi
done
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
  export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set and no saved token was found." >&2
  echo "       Expected $SCRIPT_DIR/.env, $REPO_ROOT/.env or ~/.cache/huggingface/token" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
export MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"

mkdir -p "$HF_HUB_CACHE"
AVAIL_GB=$(df -BG --output=avail "$HF_HUB_CACHE" | tail -1 | tr -dc '0-9')
echo "Model     : $MODEL"
echo "Cache dir : $HF_HUB_CACHE"
echo "Free space: ${AVAIL_GB} GB  (Llama-3.1-70B needs ~141 GB)"
if [ "$AVAIL_GB" -lt 200 ]; then
  echo
  echo "WARNING: less than 200 GB free." >&2
  echo "         Llama-3.1-70B alone needs ~141 GB." >&2
  read -r -p "Continue? [y/N] " ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "aborted."; exit 1; }
fi

# Pass executor.py the same --model id, or the cache key will not match at load time.
"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["MODEL"],
    token=os.environ["HF_TOKEN"],
    allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
    ignore_patterns=["original/*"],   # skip the duplicate .pth consolidated copy
    max_workers=8,
)
print("\nDownloaded to:", path)
PY

echo "Done."
