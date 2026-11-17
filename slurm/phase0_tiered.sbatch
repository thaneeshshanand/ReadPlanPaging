#!/bin/bash
# Phase 0 on the cluster: Qwen2.5-7B-Instruct at 8K-32K, in three stages (docs/PHASE0.md,
# Decision; the go rule and variant list are frozen until this run and the 3B laptop run are in),
# preceded by a CUDA replication of the Granite 4K laptop run.
#
#   STAGE=replicate4k  Granite 3.3 2B on the 4K continuation set (data/prompts/granite_cont_4k.jsonl),
#                  same revision and arguments as the MPS run: fp16, the bar (K4/V3), uniform
#                  4-bit (both group256, with --verify against stock SDPA), then e and f on 4/3.
#                  Then analyses it and prints a per-prompt and summary comparison against
#                  results/prescreen-4k/granite-3.3-2b. Run it unsharded (no --array).
#   STAGE=verify   one prompt: RoPE check and the patched fp16 path against stock SDPA on
#                  CUDA (attention output and teacher-forced logits); fails if they differ
#   STAGE=uniform  uniform pre-screen that sets this model's bar: fp16, uniform 4- and 3-bit,
#                  asymmetric K4/V3, K4/V2, K3/V4 (group256 keys)
#   STAGE=ladder   variants e (coldkeys_R4) and f (live) on the 4/3 and 4/2 ladders
#
# Run the stages in order. Submit from the repo root, e.g.:
#   STAGE=replicate4k sbatch --time=02:00:00 slurm/phase0_tiered.sbatch
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

STAGE="${STAGE:?set STAGE=replicate4k, verify, uniform or ladder}"
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
  replicate4k)
    GM=ibm-granite/granite-3.3-2b-instruct
    GREV=707f574c62054322f6b5b04b6d075f0a8f05e0f0          # revision of the MPS run
    GOUT="${OUT_REPLICATE:-runs/cluster/granite-3.3-2b-replicate4k}"
    REF=results/prescreen-4k/granite-3.3-2b
    [[ "$SHARD" == 0/1 ]] || { echo "replicate4k runs unsharded: submit without --array" >&2; exit 1; }
    g=(--model "$GM" --revision "$GREV" --prompts data/prompts/granite_cont_4k.jsonl --out "$GOUT"
       --max-context 4096 --max-new-tokens 129 --reference record --eps-w 1e-3 --eps-c 0.02
       --key-codecs group256 --device cuda ${LIMIT:+--limit "$LIMIT"})
    python -u phase0/tiered.py "${g[@]}" --verify --variants fp16 uniformK4V3 uniform4
    python -u phase0/tiered.py "${g[@]}" --hot-bits 4 --warm-bits 3 --variants coldkeys_R4 live
    python -u phase0/analyze_d.py --runs "$GOUT" --out "$GOUT/analysis"
    python -u phase0/compare_runs.py --a "$GOUT/analysis" --b "$REF" --labels cuda mps --out "$GOUT/analysis" \
      --variants fp16 uniformK4V3_kgroup256 uniform4_kgroup256 \
                 coldkeys_R4_w0.001_c0.02_h4b3_kgroup256 live_w0.001_c0.02_h4b3_kgroup256 ;;
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
