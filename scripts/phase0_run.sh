#!/usr/bin/env bash
# Phase 0 runner for one GPU (or the laptop), plain bash, no Slurm. Resumable: rerun the same
# command after an interruption and finished steps are skipped (step markers in $OUT/.steps/),
# while tiered.py itself skips (prompt, variant) pairs that already have results.
#
# Stages (docs/PHASE0.md, Decision; go rule and variant list frozen):
#   replicate4k  Granite 3.3 2B on the 4K continuation set at the revision of the committed MPS
#                run: fp16, the bar (K4/V3), uniform 4-bit (group256, with --verify against stock
#                SDPA), then e and f on 4/3. Analyses it and compares per prompt and per summary
#                row against $REF (default results/prescreen-4k/granite-3.3-2b), flagging prompts
#                whose perplexity change differs by more than $FLAG_PP points (default 0.3).
#   verify       Qwen2.5-7B, one prompt: patched fp16 path against stock SDPA; fails on mismatch.
#   uniform      7B uniform pre-screen that sets its bar: fp16, uniform 4/3-bit, K4/V3, K4/V2, K3/V4.
#   ladder       7B variants e (coldkeys_R4) and f (live) on the 4/3 and 4/2 ladders.
#   all          replicate4k, verify, uniform, ladder in that order.
# Several stages may be given: scripts/phase0_run.sh verify uniform ladder
#
# Environment:
#   HF_HOME   (required) Hugging Face cache, e.g. /workspace/hf on the rented GPU
#   PYTHON    interpreter (default: python); use the torch 2.15.0 env for new runs
#   DEVICE    cuda | mps | auto (default auto)
#   MODEL, PROMPTS, MODE (continuation | tasks), OUT, LIMIT    for the 7B stages, as before
#   OUT_REPLICATE, REF, FLAG_PP                                for replicate4k
#   FORCE=1   rerun steps that have a marker (tiered.py still keeps finished prompts)
#
# On a rented GPU, run it detached and reattach later, e.g.:
#   HF_HOME=/workspace/hf PYTHON=python nohup scripts/phase0_run.sh replicate4k verify > runs/run.log 2>&1 &
# data/ is not in git: copy data/prompts/ to the GPU machine first. Size the 7B stages from
# the per-prompt timings in the replicate4k and verify logs.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HF_HOME:?set HF_HOME to the Hugging Face cache directory}"
export HF_HOME TOKENIZERS_PARALLELISM=false
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PROMPTS="${PROMPTS:-data/prompts/qwen7b_cont_8k-32k.jsonl}"
MODE="${MODE:-continuation}"
OUT="${OUT:-runs/gpu/$(basename "$MODEL" | tr 'A-Z' 'a-z')-$MODE}"
LIMIT="${LIMIT:-}"
OUT_REPLICATE="${OUT_REPLICATE:-runs/gpu/granite-3.3-2b-replicate4k}"
REF="${REF:-results/prescreen-4k/granite-3.3-2b}"
FLAG_PP="${FLAG_PP:-0.3}"

(( $# )) || { sed -n '2,30p' "$0"; exit 1; }
stages=()
for s in "$@"; do
  case "$s" in
    all) stages+=(replicate4k verify uniform ladder) ;;
    replicate4k|verify|uniform|ladder) stages+=("$s") ;;
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
  base+=(--reference record --max-new-tokens 129)
else
  base+=(--reference greedy --free-run --max-new-tokens 128)
fi
tiered() { "$PYTHON" -u phase0/tiered.py --model "$MODEL" --prompts "$PROMPTS" "${base[@]}" "$@"; }
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
      FORCE=1 step replicate4k-compare "$OUT_REPLICATE" "$PYTHON" -u phase0/compare_runs.py \
        --a "$OUT_REPLICATE/analysis" --b "$REF" --labels new ref --flag-pp "$FLAG_PP" --out "$OUT_REPLICATE/analysis" \
        --variants fp16 uniformK4V3_kgroup256 uniform4_kgroup256 \
                   coldkeys_R4_w0.001_c0.02_h4b3_kgroup256 live_w0.001_c0.02_h4b3_kgroup256 ;;
    verify)
      need "$PROMPTS"
      step verify "$OUT" tiered --out "$OUT/verify" --variants fp16 uniform4 --verify --limit "${LIMIT:-1}" ;;
    uniform)
      need "$PROMPTS"
      step uniform "$OUT" tiered --out "$OUT" ${LIMIT:+--limit "$LIMIT"} \
        --variants fp16 uniform4 uniform3 uniformK4V3 uniformK4V2 uniformK3V4
      FORCE=1 step analysis "$OUT" "$PYTHON" -u phase0/analyze_d.py --runs "$OUT" --out "$OUT/analysis" ;;
    ladder)
      need "$PROMPTS"
      for warm in 3 2; do
        step "ladder-4-$warm" "$OUT" tiered --out "$OUT" ${LIMIT:+--limit "$LIMIT"} \
          --hot-bits 4 --warm-bits "$warm" --variants coldkeys_R4 live
      done
      FORCE=1 step analysis "$OUT" "$PYTHON" -u phase0/analyze_d.py --runs "$OUT" --out "$OUT/analysis" ;;
  esac
done
log "all requested stages done"
