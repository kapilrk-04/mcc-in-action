#!/usr/bin/env bash
# Run the full 4x2x2 grid for Llama-3.1-70B-Instruct on ONE worker (shard).
#
#   4 conditions x 2 schemas x 2 visibilities = 16 configs x 2038 users
#
# Usage:
#   GPU_GROUPS="0,1,2,3" ./run_grid.sh 0                  # one worker
#
#   GPU_GROUPS="0,1,2,3 4,5,6,7" ./run_grid.sh 0          # two workers, launch
#   GPU_GROUPS="0,1,2,3 4,5,6,7" ./run_grid.sh 1          # each in its own shell
#
# GPU_GROUPS is a space-separated list of comma-separated GPU ids, one group
# per worker. The argument picks which group this process uses, and the number
# of groups is the number of user shards. Every worker MUST be launched with
# the same GPU_GROUPS string, or the shards will overlap (see README.md).
#
# Each group's size becomes vLLM's tensor_parallel_size. Llama-3.1-70B has 64
# attention heads and 8 KV heads, so TP must be 1, 2, 4 or 8, and the group
# must hold the ~141 GB of bf16 weights plus KV cache.
#
# Optional environment:
#   NUM_SHARDS   default: number of groups in GPU_GROUPS
#   BATCH_SIZE   users driven concurrently per worker (default 64)
#   PYTHON_BIN   interpreter with vllm + pandas installed (default python3)
#   HF_HUB_CACHE model cache dir (default ../hf_models, same as download_model.sh)
#   LOG_DIR      default logs_70b
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
SHARD="${1:?usage: GPU_GROUPS=\"0,1,2,3\" ./run_grid.sh <shard index into GPU_GROUPS>}"

GPU_GROUPS="${GPU_GROUPS:-0,1,2,3}"
read -r -a GPU_GROUPS_ARR <<< "$GPU_GROUPS"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_GROUPS_ARR[@]}}"

if [ "$SHARD" -ge "${#GPU_GROUPS_ARR[@]}" ]; then
  echo "shard $SHARD but only ${#GPU_GROUPS_ARR[@]} GPU group(s) in GPU_GROUPS=\"$GPU_GROUPS\"" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_GROUPS_ARR[$SHARD]}"
TP=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")

# Fail here rather than during engine init.
case "$TP" in
  1|2|4|8) ;;
  *) echo "FATAL: TP=$TP (from GPUs $CUDA_VISIBLE_DEVICES) is not usable." >&2
     echo "       Llama-3.1-70B has 64 attention heads and 8 KV heads, so vLLM" >&2
     echo "       requires TP in {1,2,4,8}. Use groups of 1, 2, 4 or 8 GPUs." >&2
     exit 1 ;;
esac

# HF_TOKEN etc. may live in a .env next to the project or one level up.
for f in .env ../.env; do [ -f "$f" ] && { set -a; source "$f"; set +a; }; done

PYTHON_BIN="${PYTHON_BIN:-python3}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$(cd .. && pwd)/hf_models}"
export VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false

# vLLM contacts huggingface.co at every engine start even when the weights are
# cached, and fails if the network is down. Once the model is downloaded, run
# offline.
if [ -z "${HF_HUB_OFFLINE:-}" ] && \
   ls "$HF_HUB_CACHE"/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/*/config.json >/dev/null 2>&1; then
  export HF_HUB_OFFLINE=1
fi

# Users driven concurrently; raise it if there is spare KV cache.
BATCH_SIZE="${BATCH_SIZE:-64}"
LOG_DIR="${LOG_DIR:-logs_70b}"; mkdir -p "$LOG_DIR"

"$PYTHON_BIN" -c "import vllm,pandas" 2>/dev/null || {
  echo "FATAL: $PYTHON_BIN cannot import vllm/pandas (set PYTHON_BIN)" >&2; exit 1; }

echo "llama-70B grid | shard $SHARD/$NUM_SHARDS | GPUs $CUDA_VISIBLE_DEVICES | TP=$TP | batch $BATCH_SIZE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
echo "16 configs, ${NUM_SHARDS}-way user sharding"
echo

for cond in none demographics_only context_only demographics_context; do
  for schema in 1 2; do
    for vis in seen unseen; do
      name="${cond}_s${schema}_${vis}_shard${SHARD}"
      log="${LOG_DIR}/${name}.log"
      echo "=== [$(date '+%F %T')] start $name"
      rc=0
      "$PYTHON_BIN" executor.py \
        --condition "$cond" --schema "$schema" --movie_visibility "$vis" \
        --tensor_parallel_size "$TP" --batch_size "$BATCH_SIZE" \
        --shard "$SHARD" --num_shards "$NUM_SHARDS" \
        >> "$log" 2>&1 || rc=$?
      if [ "$rc" -eq 0 ]; then echo "=== [$(date '+%F %T')] done  $name"
      else echo "=== [$(date '+%F %T')] FAILED $name (exit $rc)"; tail -5 "$log" | sed 's/^/      /'; fi
    done
  done
done
echo "all 16 configs finished."
