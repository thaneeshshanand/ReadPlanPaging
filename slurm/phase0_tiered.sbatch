#!/bin/bash
# Phase 0 part D on the cluster: tiered-read simulation with the reduced variant list
#   fp16, fp8, uniform4, a (fresh), c (promote_R4), d (twotier_R4), e (coldkeys_R4)
# on a HOT/WARM ladder (HOT_BITS/WARM_BITS) with KEY_CODECS key scales. fp8 here is the
# emulation; reproduce it with vLLM's real fp8 KV cache before quoting it.
# A second ladder is a second submission into the same OUT (variants are added to
# finished prompts), e.g. WARM_BITS=2 sbatch --array=0-9 slurm/phase0_tiered.sbatch
#
# One array task per prompt shard. Submit from the repo root, e.g.:
#   sbatch --array=0-9 slurm/phase0_tiered.sbatch                                  # continuation set (perplexity)
#   PROMPTS=data/prompts/qwen7b_tasks_8k-32k.jsonl MODE=tasks sbatch --array=0-9 slurm/phase0_tiered.sbatch
#
# First submit a one-prompt timing job and size --time from it (none of these numbers
# have been measured on a GPU yet):
#   LIMIT=1 sbatch --array=0 --time=01:00:00 slurm/phase0_tiered.sbatch
#
# Static key codec: run slurm/phase0_calibrate.sbatch first (writes $CALIB).
# data/ is not in git: copy data/prompts/ to the cluster checkout before submitting.
# Analyse afterwards (CPU is fine):
#   python phase0/analyze_d.py --runs "$OUT" --out "$OUT/analysis"
#
#SBATCH --job-name=rpp-tiered
#SBATCH --partition=gpu            # check partition and GPU names with: sinfo -o "%P %G"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00            # placeholder until the timing job has run
#SBATCH --output=runs/slurm/%x-%A_%a.out

set -euo pipefail

# ---- setup: adjust to the cluster environment -------------------------------------
# module load cuda/12.1            # if the cluster needs it
# source "$HOME/venvs/rpp/bin/activate"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export TOKENIZERS_PARALLELISM=false
# -------------------------------------------------------------------------------------

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PROMPTS="${PROMPTS:-data/prompts/qwen7b_cont_8k-32k.jsonl}"
MODE="${MODE:-continuation}"                      # continuation | tasks
KEY_CODECS="${KEY_CODECS:-group256}"              # cheapest passing uniform codec in the 1.5B MPS pre-screen
HOT_BITS="${HOT_BITS:-4}"                         # ladder HOT width (the bar's width); 16 = fp16
WARM_BITS="${WARM_BITS:-3}"                       # ladder WARM width: 3 or 2 (see the MPS ladder run)
CALIB="${CALIB:-runs/calib/$(basename "$MODEL" | tr 'A-Z' 'a-z').pt}"
OUT="${OUT:-runs/cluster/$(basename "$MODEL" | tr 'A-Z' 'a-z')-$MODE}"
LIMIT="${LIMIT:-}"
VARIANTS="fp16 fp8 uniform4 fresh promote_R4 twotier_R4 coldkeys_R4"

mkdir -p "$OUT" runs/slurm
SHARD="${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"

args=(--model "$MODEL" --prompts "$PROMPTS" --out "$OUT" --max-context 32000
      --variants $VARIANTS --key-codecs $KEY_CODECS --hot-bits "$HOT_BITS" --warm-bits "$WARM_BITS"
      --eps-w 1e-3 --eps-c 0.02 --shard "$SHARD" --device cuda)
if [[ " $KEY_CODECS " == *" static "* ]]; then
  [[ -f "$CALIB" ]] || { echo "missing calibration $CALIB; run slurm/phase0_calibrate.sbatch" >&2; exit 1; }
  args+=(--key-calib "$CALIB")
fi
if [[ "$MODE" == continuation ]]; then
  args+=(--reference record --max-new-tokens 129)
else
  args+=(--reference greedy --free-run --max-new-tokens 128)
fi
[[ -n "$LIMIT" ]] && args+=(--limit "$LIMIT")

echo "job=${SLURM_JOB_ID:-local} shard=$SHARD host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python phase0/tiered.py "${args[@]}"
