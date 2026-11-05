#!/usr/bin/env bash
#SBATCH --job-name=vaxtron_models
#SBATCH --partition=cpu_r6625
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=vaxtron_setup-%j.out
#SBATCH --error=vaxtron_setup-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f setup_vaxtron_runtime.sh ]]; then
  echo "ERROR: Submit this job from the paper-extracting repository root."
  exit 1
fi

module purge
module load Python/3.10.4-GCCcore-11.3.0

echo "=== VERIFY EXISTING LLAMA-SERVER ==="
test -x .runtime/llama.cpp/build/bin/llama-server

echo "=== INSTALL PINNED MODELS AND PYTHON DEPENDENCIES ==="
export BUILD_LLAMA=0
export PYTHON_BIN=python3

bash setup_vaxtron_runtime.sh

echo "=== INSTALLATION COMPLETED ==="
