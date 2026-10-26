#!/bin/bash
# Launcher for the TFT BTC/USDT project on a Slurm cluster.
#
# Usage:
#   ./run.sh setup            create the Python venv and install dependencies (login node)
#   ./run.sh check            look-ahead + timestamp-convention checks (run once, before training)
#   ./run.sh test             quick 2-epoch interactive GPU test
#   ./run.sh submit           submit the full training job with sbatch
#   ./run.sh resume <rundir>  continue an interrupted run, e.g. ./run.sh resume runs/tft_123456
#   ./run.sh status | logs | cancel
#
# Extra training flags can be passed through TRAIN_ARGS, e.g.
#   TRAIN_ARGS="--no-htf --norm global" ./run.sh submit
# Other env vars: PARTITION=gpu GRES=gpu:1 CSV_15M=... CSV_1H=... CSV_4H=... VENV=$HOME/venvs/tft

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PARTITION="${PARTITION:-gpu}"      # CHECK: confirm partition name in the Tufts HPC docs
GRES="${GRES:-gpu:1}"              # CHECK: some clusters need a type, e.g. gpu:a100:1
export CSV_15M="${CSV_15M:-data/BTCUSDT_15m.csv}"
export CSV_1H="${CSV_1H:-data/BTCUSDT_1h.csv}"
export CSV_4H="${CSV_4H:-data/BTCUSDT_4h.csv}"
VENV="${VENV:-$HOME/venvs/tft}"
JOB_NAME="tft_btc"
DATA_ARGS="--csv_15m $CSV_15M --csv_1h $CSV_1H --csv_4h $CSV_4H"

# module purge && module load miniforge cuda   # CHECK `module avail`, then uncomment

need_files() {
    for f in src/train.py src/data.py src/features.py src/model.py src/loader.py src/signals.py \
             src/backtest.py src/config.py src/check_leakage.py \
             train_tft.sbatch requirements.txt; do
        [ -f "$f" ] || { echo "Missing $f in $(pwd)"; exit 1; }
    done
    for f in "$CSV_15M" "$CSV_1H" "$CSV_4H"; do
        [ -f "$f" ] || { echo "Missing data file: $f"; exit 1; }
    done
    mkdir -p logs runs
}

setup_env() {
    if [ ! -d "$VENV" ]; then
        echo "Creating venv at $VENV ..."
        python3 -m venv "$VENV"
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        source "$VENV/bin/activate"
        python -c "import ta" 2>/dev/null || pip install ta
        echo "Venv ready at $VENV"
    fi
}

case "${1:-help}" in
    setup)
        setup_env
        ;;
    check)
        need_files; setup_env
        # shellcheck disable=SC2086
        python src/check_leakage.py $DATA_ARGS ${TRAIN_ARGS:-}
        ;;
    test)
        need_files; setup_env
        echo "Starting 2-epoch smoke test on a GPU node (partition=$PARTITION) ..."
        srun --partition="$PARTITION" --gres="$GRES" --cpus-per-task=4 --mem=16G --time=00:30:00 \
             --job-name="${JOB_NAME}_test" --pty bash -c "
                source '$VENV/bin/activate' &&
                nvidia-smi -L &&
                python src/train.py $DATA_ARGS --out runs/smoke_test --epochs 2 ${TRAIN_ARGS:-}
             "
        ;;
    submit)
        need_files; setup_env
        JID=$(sbatch --parsable --partition="$PARTITION" --gres="$GRES" --export=ALL train_tft.sbatch)
        echo "Submitted job $JID"
        echo "  status : ./run.sh status"
        echo "  logs   : ./run.sh logs   (or tail -f logs/${JOB_NAME}_${JID}.out)"
        echo "  output : runs/tft_${JID}/"
        ;;
    resume)
        RUN="${2:?usage: ./run.sh resume <rundir>}"
        [ -f "$RUN/last.pt" ] || { echo "No $RUN/last.pt to resume from"; exit 1; }
        need_files; setup_env
        JID=$(OUT_DIR="$RUN" sbatch --parsable --partition="$PARTITION" --gres="$GRES" --export=ALL train_tft.sbatch)
        echo "Resubmitted as job $JID, continuing in $RUN"
        ;;
    status)
        squeue -u "$USER" -o "%.10i %.12P %.12j %.8T %.10M %.10l %R"
        ;;
    logs)
        LOG=$(ls -t logs/${JOB_NAME}_*.out 2>/dev/null | head -n1 || true)
        [ -n "$LOG" ] || { echo "No logs yet in logs/"; exit 1; }
        echo "Following $LOG (Ctrl-C to stop)"
        tail -f "$LOG"
        ;;
    cancel)
        scancel -u "$USER" --name="$JOB_NAME" && echo "Cancelled $JOB_NAME jobs"
        ;;
    *)
        sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
        ;;
esac
