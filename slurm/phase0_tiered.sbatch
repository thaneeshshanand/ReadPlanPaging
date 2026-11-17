#!/bin/bash
# Slurm wrapper for scripts/phase0_run.sh, which holds the stages and is the primary runner
# (Explorer is not available for this project; the GPU runs use the plain-bash runner over SSH).
# Kept for a Slurm cluster, should one become available:
#   STAGES="replicate4k verify" HF_HOME=/scratch/$USER/hf sbatch slurm/phase0_tiered.sbatch
#SBATCH --job-name=rpp-phase0
#SBATCH --partition=gpu            # check partition and GPU names with: sinfo -o "%P %G"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00            # placeholder: size from the replicate4k and verify logs
#SBATCH --output=runs/slurm/%x-%j.out
set -euo pipefail
# module load cuda/12.x; source "$HOME/venvs/rpp/bin/activate"   # adjust to the cluster
read -r -a stages <<< "${STAGES:?set STAGES, e.g. STAGES=\"replicate4k verify\"}"
exec scripts/phase0_run.sh "${stages[@]}"
