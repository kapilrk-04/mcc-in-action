#!/usr/bin/env bash
# Ablation runner for executor_gemini.py
#
# One invocation runs every selected ablation for a schema through the Gemini
# Batch Jobs API. The executor sweeps both movie visibilities (unseen | seen)
# internally, so each condition yields two runs.
#
# Conditions (what the model is shown):
#   demographics_only    : demographics,    no history
#   context_only         : no demographics, history
#   demographics_context : demographics,    history
#   none                 : no demographics, no history  (Q&A from EXPLORE only)
#
# Fixed values
#   Temperature : 1.0
#   N_history   : 10       (history conditions only)
#   Nature      : diverse  (history conditions only)
#
# The Gemini API key is read from GEMINI_API_KEY in config.json.
# Set PYTHON to override the interpreter (default: python).
#
# Usage:
#   bash run_ablations_gemini.sh                                  # schema 1, flash, all conditions
#   bash run_ablations_gemini.sh --schema 2                       # schema 2
#   bash run_ablations_gemini.sh --model pro --thinking medium    # pro with thinking
#   bash run_ablations_gemini.sh --condition context_only         # single condition
#   bash run_ablations_gemini.sh --poll_interval 60               # poll every 60s
#   bash run_ablations_gemini.sh --rerun                          # rerun everything from turn 1

set -euo pipefail

PYTHON="${PYTHON:-python}"

SCHEMA=1
MODEL=flash
THINKING=none
CONDITION=""
POLL_INTERVAL=30
RERUN=false

TEMPERATURE=1.0
N_HISTORY=10
NATURE=diverse

ALL_CONDITIONS=(demographics_only context_only demographics_context none)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --schema)        SCHEMA="$2";        shift 2 ;;
        --model)         MODEL="$2";         shift 2 ;;
        --thinking)      THINKING="$2";      shift 2 ;;
        --condition)     CONDITION="$2";     shift 2 ;;
        --poll_interval) POLL_INTERVAL="$2"; shift 2 ;;
        --rerun)         RERUN=true;         shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$SCHEMA" != "1" && "$SCHEMA" != "2" ]]; then
    echo "Error: --schema must be 1 or 2"; exit 1
fi
if [[ "$MODEL" != "flash" && "$MODEL" != "pro" ]]; then
    echo "Error: --model must be flash or pro"; exit 1
fi
if [[ "$THINKING" != "none" && "$THINKING" != "medium" ]]; then
    echo "Error: --thinking must be none or medium"; exit 1
fi
if [[ -n "$CONDITION" && ! " ${ALL_CONDITIONS[*]} " =~ " ${CONDITION} " ]]; then
    echo "Error: --condition must be one of: ${ALL_CONDITIONS[*]}"; exit 1
fi

if [[ "$MODEL" == "flash" ]]; then
    MODEL_FULL="gemini-2.5-flash"
    MODEL_SUFFIX="flash"
    OUT_DIR_MODEL="gemini_25_flash"
else
    MODEL_FULL="gemini-2.5-pro"
    MODEL_SUFFIX="pro_think_${THINKING}"
    if [[ "$THINKING" == "medium" ]]; then
        OUT_DIR_MODEL="gemini_25_pro_thinking"
    else
        OUT_DIR_MODEL="gemini_25_pro_no_thinking"
    fi
fi

OUT_ROOT="rl_explore_exploit_results/${OUT_DIR_MODEL}"
LOG_DIR="ablation_logs/${OUT_DIR_MODEL}"
mkdir -p "$LOG_DIR"

if [[ -n "$CONDITION" ]]; then
    CONDITIONS=("$CONDITION")
    LOG_PATH="${LOG_DIR}/schema_batch_schema${SCHEMA}_${CONDITION}_${MODEL_SUFFIX}.log"
else
    CONDITIONS=("${ALL_CONDITIONS[@]}")
    LOG_PATH="${LOG_DIR}/schema_batch_schema${SCHEMA}_${MODEL_SUFFIX}.log"
fi

args=(executor_gemini.py
    --schema        "$SCHEMA"
    --model         "$MODEL_FULL"
    --thinking      "$THINKING"
    --temperature   "$TEMPERATURE"
    --n_history     "$N_HISTORY"
    --nature        "$NATURE"
    --poll_interval "$POLL_INTERVAL"
    --conditions    "${CONDITIONS[@]}"
)
[[ "$RERUN" == "true" ]] && args+=(--rerun)

echo "======================================================================"
echo "ABLATION STUDY  –  schema-batch  (${MODEL_FULL}, thinking=${THINKING})"
echo "======================================================================"
printf "Python       : %s  (%s)\n" "$PYTHON" "$("$PYTHON" --version 2>&1)"
printf "Schema       : %s\n" "$SCHEMA"
printf "Conditions   : %s\n" "${CONDITIONS[*]}"
printf "Visibilities : unseen seen\n"
printf "Total runs   : %d\n" $(( ${#CONDITIONS[@]} * 2 ))
printf "Temperature  : %s\n" "$TEMPERATURE"
printf "N_history    : %s  (history conditions only)\n" "$N_HISTORY"
printf "Nature       : %s  (history conditions only)\n" "$NATURE"
printf "Poll interval: %ss\n" "$POLL_INTERVAL"
printf "Rerun        : %s\n" "$RERUN"
printf "Results dir  : %s\n" "$OUT_ROOT"
printf "Log          : %s\n" "$LOG_PATH"
echo "======================================================================"

start=$(date +%s)
set +e
"$PYTHON" -u "${args[@]}" > "$LOG_PATH" 2>&1
rc=$?
set -e
elapsed=$(( $(date +%s) - start ))

if [[ $rc -eq 0 ]]; then
    echo "Done: OK  (${elapsed}s)"
else
    echo "Done: FAILED (rc=${rc}, ${elapsed}s) – see ${LOG_PATH}"
fi
exit $rc
