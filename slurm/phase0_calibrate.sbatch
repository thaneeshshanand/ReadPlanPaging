#!/bin/bash
# Calibrate static key quantiser ranges (needed before KEY_CODECS includes "static").
#   sbatch slurm/phase0_calibrate.sbatch
#
#SBATCH --job-name=rpp-calib
#SBATCH --partition=gpu            # check with: sinfo -o "%P %G"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
# module load cuda/12.1
# source "$HOME/venvs/rpp/bin/activate"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CALIB_PROMPTS="${CALIB_PROMPTS:-data/prompts/qwen7b_cont_8k-32k.calib.jsonl}"
CALIB="${CALIB:-runs/calib/$(basename "$MODEL" | tr 'A-Z' 'a-z').pt}"
mkdir -p runs/slurm "$(dirname "$CALIB")"
python phase0/calibrate_keys.py --model "$MODEL" --prompts "$CALIB_PROMPTS" --out "$CALIB" \
  --max-context 8192 --device cuda
