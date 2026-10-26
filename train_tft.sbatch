#!/bin/bash
#SBATCH --job-name=tft_btc
#SBATCH --partition=gpu            # CHECK: confirm GPU partition name in the Tufts HPC docs / `sinfo`
#SBATCH --gres=gpu:1               # CHECK: some clusters want a type, e.g. gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs runs

# ---- environment (CHECK `module avail` for exact names on Tufts, then uncomment) ----
# module purge
# module load miniforge cuda
VENV="${VENV:-$HOME/venvs/tft}"
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$VENV/bin/activate"
    python -c "import ta" 2>/dev/null || pip install ta
fi

# Output folder: new per job by default; set OUT_DIR=runs/tft_<oldid> to RESUME an earlier run
OUT_DIR="${OUT_DIR:-runs/tft_${SLURM_JOB_ID}}"

nvidia-smi
echo "Job $SLURM_JOB_ID on $(hostname) started $(date) -> $OUT_DIR"

python src/train.py \
    --csv_15m "${CSV_15M:-data/BTCUSDT_15m.csv}" \
    --csv_1h  "${CSV_1H:-data/BTCUSDT_1h.csv}" \
    --csv_4h  "${CSV_4H:-data/BTCUSDT_4h.csv}" \
    --out "$OUT_DIR" \
    --resume \
    ${TRAIN_ARGS:-}

echo "Finished $(date)"
