#!/usr/bin/env bash
# Phase 0 runner for one GPU (or the laptop), plain bash, no Slurm. Resumable: rerun the same
# command after an interruption and finished steps are skipped (step markers in $OUT/.steps/),
# while tiered.py itself skips (prompt, variant) pairs that already have results.
#
# Stages, in this order on the GPU (docs/PHASE0.md, Decision; go rule and variant list frozen).
# Inside each 7B stage the context lengths run 8K, then 16K, then 32K ($LENGTHS), one step each.
#   replicate4k  Granite 3.3 2B on the 4K continuation set at the revision of the committed MPS
#                run: fp16, the bar (K4/V3), uniform 4-bit (group256, with --verify against stock
#                SDPA), then e and f on 4/3. Analyses it and checks it against $REF (default
#                results/prescreen-4k/granite-3.3-2b): PASS if the verdict is unchanged, every
#                pooled perplexity change is within $POOLED_PP points (0.25) and no prompt differs
#                by more than $PROMPT_PP points (5). Per-prompt drift is reported as a distribution.
#                The runner stops on FAIL.
#   verify       Qwen2.5-7B, one prompt per length: patched fp16 path against stock SDPA.
#   timing       $TIMING_N prompts per length (fp16 + uniform4, then e and f on 4/3) into $OUT/timing;
#                prints the estimated hours of the uniform and ladder stages.
#   uniform      7B uniform pre-screen that sets its bar: fp16, uniform 4/3-bit, K4/V3, K4/V2, K3/V4.
#   ladder       7B variants e (coldkeys_R4) and f (live) on the 4/3 and 4/2 ladders.
#   all          replicate4k, verify, timing, uniform, ladder.
# uniform and ladder first check the timing estimate of the work still to do and stop if it is
# over $MAX_HOURS (default 1): confirm the estimate, then rerun with a larger MAX_HOURS.
# Several stages may be given: scripts/phase0_run.sh verify timing
#
# Environment:
#   HF_HOME   (required) Hugging Face cache, e.g. /workspace/hf on the rented GPU
#   PYTHON    interpreter (default: python); use the torch 2.15.0 env (requirements.txt)
#   DEVICE    cuda | mps | auto (default auto)
#   MODEL, PROMPTS, MODE (continuation | tasks), OUT, LIMIT    for the 7B stages
#             (continuation: 257 reference tokens, 256 scored; tasks mode runs without the length split)
#   LENGTHS   context lengths in order (default "8192 16384 32000")
#   TIMING_N  prompts per length in the timing stage (default 1); MAX_HOURS (default 1)
#   OUT_REPLICATE, REF, POOLED_PP, PROMPT_PP                   for replicate4k
#   FORCE=1   rerun steps that have a marker (tiered.py still keeps finished prompts)
#
# On a rented GPU, run it detached and reattach later, e.g.:
#   HF_HOME=/workspace/hf nohup scripts/phase0_run.sh replicate4k verify timing > runs/gpu.log 2>&1 &
# data/ is not in git: copy data/prompts/ to the GPU machine first.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HF_HOME:?set HF_HOME to the Hugging Face cache directory}"
export HF_HOME TOKENIZERS_PARALLELISM=false
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PROMPTS="${PROMPTS:-data/prompts/qwen7b_cont_8k-32k_ref257.jsonl}"
MODE="${MODE:-continuation}"
OUT="${OUT:-runs/gpu/$(basename "$MODEL" | tr 'A-Z' 'a-z')-$MODE}"
LIMIT="${LIMIT:-}"
OUT_REPLICATE="${OUT_REPLICATE:-runs/gpu/granite-3.3-2b-replicate4k}"
REF="${REF:-results/prescreen-4k/granite-3.3-2b}"
POOLED_PP="${POOLED_PP:-0.25}"
PROMPT_PP="${PROMPT_PP:-5}"
LENGTHS="${LENGTHS:-8192 16384 32000}"
TIMING_N="${TIMING_N:-1}"
MAX_HOURS="${MAX_HOURS:-1}"
read -r -a lengths <<< "$LENGTHS"
[[ "$MODE" == continuation ]] || lengths=(all)

(( $# )) || { sed -n '2,39p' "$0"; exit 1; }
stages=()
for s in "$@"; do
  case "$s" in
    all) stages+=(replicate4k verify timing uniform ladder) ;;
    replicate4k|verify|timing|uniform|ladder) stages+=("$s") ;;
    *) echo "unknown stage: $s" >&2; exit 1 ;;
  esac
done

log() { echo "[$(date '+%F %T')] $*"; }

# step NAME DIR CMD...: run CMD unless DIR/.steps/NAME exists; log to DIR/logs/NAME.log
step() {
  local name=$1 dir=$2; shift 2
  local marker="$dir/.steps/$name" logf="$dir/logs/$name.log"
  mkdir -p "$dir/.steps" "$dir/logs"
  if [[ -e "$marker" && "${FORCE:-0}" != 1 ]]; then
    log "skip $name (done $(cat "$marker"))"; return 0
  fi
  log "start $name -> $logf"
  if "$@" > >(tee -a "$logf") 2>&1; then
    date '+%F %T' > "$marker"; log "done $name"
  else
    local rc=$?; log "FAILED $name (exit $rc), see $logf"; exit "$rc"
  fi
}

log "stages=${stages[*]} commit=$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- phase0 || echo '+dirty') host=$(hostname)"
"$PYTHON" -c "import sys, torch, transformers; print('python', sys.version.split()[0], sys.executable); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__); print('cuda available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', '| mps available', torch.backends.mps.is_available())"
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

base=(--max-context 32000 --key-codecs group256 --eps-w 1e-3 --eps-c 0.02 --device "$DEVICE")
if [[ "$MODE" == continuation ]]; then
  base+=(--reference record --max-new-tokens 257)   # 256 scored tokens per prompt (docs/PHASE0.md)
else
  base+=(--reference greedy --free-run --max-new-tokens 128)
fi
tiered() { "$PYTHON" -u phase0/tiered.py --model "$MODEL" --prompts "$PROMPTS" "${base[@]}" "$@"; }
# lenarg L: tiered.py arguments selecting context length L ("all": no selection)
lenarg() { [[ "$1" == all ]] || echo "--n-tokens $1"; }
gate() {  # gate STAGE: stop if the timing estimate of the remaining work exceeds MAX_HOURS
  FORCE=1 step "$1-gate" "$OUT" "$PYTHON" -u phase0/estimate_time.py --timing "$OUT/timing" --prompts "$PROMPTS" \
    --out-dir "$OUT" --gate "$1" --max-hours "$MAX_HOURS"
}
need() { [[ -f "$1" ]] || { echo "missing $1 (data/ is not in git: copy data/prompts/ here)" >&2; exit 1; }; }

for stage in "${stages[@]}"; do
  case "$stage" in
    replicate4k)
      GM=ibm-granite/granite-3.3-2b-instruct
      GREV=707f574c62054322f6b5b04b6d075f0a8f05e0f0   # revision of the committed MPS run
      need data/prompts/granite_cont_4k.jsonl
      g=(--model "$GM" --revision "$GREV" --prompts data/prompts/granite_cont_4k.jsonl --out "$OUT_REPLICATE"
         --max-context 4096 --max-new-tokens 129 --reference record --eps-w 1e-3 --eps-c 0.02
         --key-codecs group256 --device "$DEVICE" ${LIMIT:+--limit "$LIMIT"})
      step replicate4k-uniform "$OUT_REPLICATE" "$PYTHON" -u phase0/tiered.py "${g[@]}" --verify \
        --variants fp16 uniformK4V3 uniform4
      step replicate4k-ladder "$OUT_REPLICATE" "$PYTHON" -u phase0/tiered.py "${g[@]}" --hot-bits 4 --warm-bits 3 \
        --variants coldkeys_R4 live
      FORCE=1 step replicate4k-analysis "$OUT_REPLICATE" "$PYTHON" -u phase0/analyze_d.py \
        --runs "$OUT_REPLICATE" --out "$OUT_REPLICATE/analysis"
      FORCE=1 step replicate4k-check "$OUT_REPLICATE" "$PYTHON" -u phase0/compare_runs.py --check \
        --a "$OUT_REPLICATE/analysis" --b "$REF" --labels new ref --pooled-pp "$POOLED_PP" --prompt-pp "$PROMPT_PP" \
        --out "$OUT_REPLICATE/analysis" \
        --variants fp16 uniformK4V3_kgroup256 uniform4_kgroup256 \
                   coldkeys_R4_w0.001_c0.02_h4b3_kgroup256 live_w0.001_c0.02_h4b3_kgroup256 ;;
    verify)
      need "$PROMPTS"
      for L in "${lengths[@]}"; do
        step "verify-$L" "$OUT" tiered --out "$OUT/verify" $(lenarg "$L") --limit "${LIMIT:-1}" \
          --variants fp16 uniform4 --verify
      done ;;
    timing)
      need "$PROMPTS"
      for L in "${lengths[@]}"; do
        step "timing-$L-uniform" "$OUT" tiered --out "$OUT/timing" $(lenarg "$L") --limit "$TIMING_N" \
          --variants fp16 uniform4
        step "timing-$L-ladder" "$OUT" tiered --out "$OUT/timing" $(lenarg "$L") --limit "$TIMING_N" \
          --hot-bits 4 --warm-bits 3 --variants coldkeys_R4 live
      done
      FORCE=1 step timing-estimate "$OUT" "$PYTHON" -u phase0/estimate_time.py --timing "$OUT/timing" \
        --prompts "$PROMPTS" --out-dir "$OUT" ;;
    uniform)
      need "$PROMPTS"
      gate uniform
      for L in "${lengths[@]}"; do
        step "uniform-$L" "$OUT" tiered --out "$OUT" $(lenarg "$L") ${LIMIT:+--limit "$LIMIT"} \
          --variants fp16 uniform4 uniform3 uniformK4V3 uniformK4V2 uniformK3V4
      done
      FORCE=1 step analysis "$OUT" "$PYTHON" -u phase0/analyze_d.py --runs "$OUT" --out "$OUT/analysis" ;;
    ladder)
      need "$PROMPTS"
      gate ladder
      for L in "${lengths[@]}"; do
        for warm in 3 2; do
          step "ladder-$L-4-$warm" "$OUT" tiered --out "$OUT" $(lenarg "$L") ${LIMIT:+--limit "$LIMIT"} \
            --hot-bits 4 --warm-bits "$warm" --variants coldkeys_R4 live
        done
      done
      FORCE=1 step analysis "$OUT" "$PYTHON" -u phase0/analyze_d.py --runs "$OUT" --out "$OUT/analysis" ;;
  esac
done
log "all requested stages done"
