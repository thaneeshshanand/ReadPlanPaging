#!/bin/bash
# Phase 0 on the cluster: Qwen2.5-7B-Instruct at 8K-32K, in three stages (docs/PHASE0.md,
# Decision; the go rule and variant list are frozen until this run and the 3B laptop run are in).
#
#   STAGE=verify   one prompt: RoPE check and the patched fp16 path against stock SDPA on
#                  CUDA (attention output and teacher-forced logits); fails if they differ
#   STAGE=uniform  uniform pre-screen that sets this model's bar: fp16, uniform 4- and 3-bit,
#                  asymmetric K4/V3, K4/V2, K3/V4 (group256 keys)
#   STAGE=ladder   variants e (coldkeys_R4) and f (live) on the 4/3 and 4/2 ladders
#
# Run the stages in order. Submit from the repo root, e.g.:
#   STAGE=verify  sbatch --array=0 --time=01:00:00 slurm/phase0_tiered.sbatch
#   STAGE=uniform sbatch --array=0-9 slurm/phase0_tiered.sbatch
#   STAGE=ladder  sbatch --array=0-9 slurm/phase0_tiered.sbatch
# and repeat the uniform and ladder stages on the task set for free-running task scores:
#   STAGE=uniform PROMPTS=data/prompts/qwen7b_tasks_8k-32k.jsonl MODE=tasks sbatch --array=0-9 slurm/phase0_tiered.sbatch
#   STAGE=ladder  PROMPTS=data/prompts/qwen7b_tasks_8k-32k.jsonl MODE=tasks sbatch --array=0-9 slurm/phase0_tiered.sbatch
#
# The verify job's log shows per-prompt timings; size --time for the other stages from it
# (no GPU timing has been measured yet). Stages write into the same OUT; variants are added
# to finished prompts. data/ is not in git: copy data/prompts/ to the cluster checkout first.
# Analyse afterwards (CPU is fine); by_length.csv has HOT/WARM/COLD fractions by context length:
#   python phase0/analyze_d.py --runs "$OUT" --out "$OUT/analysis"
#
#SBATCH --job-name=rpp-tiered
#SBATCH --partition=gpu            # check partition and GPU names with: sinfo -o "%P %G"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00            # placeholder until the verify job has run
#SBATCH --output=runs/slurm/%x-%A_%a.out

set -euo pipefail

# ---- setup: adjust to the cluster environment -------------------------------------
# module load cuda/12.1            # if the cluster needs it
# source "$HOME/venvs/rpp/bin/activate"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export TOKENIZERS_PARALLELISM=false
# -------------------------------------------------------------------------------------

STAGE="${STAGE:?set STAGE=verify, uniform or ladder}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PROMPTS="${PROMPTS:-data/prompts/qwen7b_cont_8k-32k.jsonl}"
MODE="${MODE:-continuation}"                      # continuation (perplexity) | tasks (free-run task scores)
OUT="${OUT:-runs/cluster/$(basename "$MODEL" | tr 'A-Z' 'a-z')-$MODE}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUT" runs/slurm
SHARD="${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"

args=(--model "$MODEL" --prompts "$PROMPTS" --max-context 32000 --key-codecs group256
      --eps-w 1e-3 --eps-c 0.02 --shard "$SHARD" --device cuda)
if [[ "$MODE" == continuation ]]; then
  args+=(--reference record --max-new-tokens 129)
else
  args+=(--reference greedy --free-run --max-new-tokens 128)
fi

echo "job=${SLURM_JOB_ID:-local} stage=$STAGE shard=$SHARD host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "$STAGE" in
  verify)
    python phase0/tiered.py "${args[@]}" --out "$OUT/verify" --variants fp16 uniform4 --verify --limit "${LIMIT:-1}" ;;
  uniform)
    python phase0/tiered.py "${args[@]}" --out "$OUT" ${LIMIT:+--limit "$LIMIT"} \
      --variants fp16 uniform4 uniform3 uniformK4V3 uniformK4V2 uniformK3V4 ;;
  ladder)
    for warm in 3 2; do
      python phase0/tiered.py "${args[@]}" --out "$OUT" ${LIMIT:+--limit "$LIMIT"} \
        --hot-bits 4 --warm-bits "$warm" --variants coldkeys_R4 live
    done ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 1 ;;
esac
