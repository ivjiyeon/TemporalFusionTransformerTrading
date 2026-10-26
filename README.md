# TemporalFusionTransformerTrading

A Temporal Fusion Transformer (TFT) classifier that turns multi-timeframe
BTC/USDT price data into long/short trading signals, with an event backtest.
Compact, pure PyTorch — no heavy TFT library required.

## What it does

1. **Load and align** 15m, 1h, and 4h BTC/USDT OHLCV data without look-ahead.
2. **Label** each 15m bar with a triple-barrier outcome over an 8h horizon:
   neutral / long / short.
3. **Train** a compact TFT on 24h input windows (96 x 15m bars) to predict
   those three classes.
4. **Generate signals** from class probabilities with a one-position-at-a-time
   state machine (enter, exit on fading conviction or a flip, forced exit
   after a max hold).
5. **Backtest** on the held-out test period, filling at the next bar's open
   with fees and slippage, and reporting return, Sharpe, max drawdown,
   win rate, and trade count.

## Layout

```
run.sh               Slurm launcher: setup / check / test / submit / resume / status / logs / cancel
train_tft.sbatch     Slurm job script (1 GPU, 4h wall time)
requirements.txt     torch, numpy, pandas, ta
src/
  config.py          All settings, also exposed as train.py CLI flags
  data.py            Load 15m/1h/4h, align higher timeframes, build features, labels, splits
  features.py        TA / price features for one timeframe
  loader.py          GPU window batching + per-window normalisation
  model.py           Temporal Fusion Transformer (variable selection, LSTM encoder,
                     gated residual networks, causal self-attention, 3-class head)
  signals.py         Probabilities -> LONG_ENTRY / SHORT_ENTRY / EXIT
  backtest.py        Event backtest: fill at next open, fees, slippage
  train.py           Train, predict, generate signals, backtest on test period
  check_leakage.py   Causality + timestamp-convention checks (run before training)
```

The Python modules import each other as flat modules (`from data import ...`).
Run them from the repo root as `python src/train.py ...` — Python puts the
script's own directory on the path, so the flat imports resolve — or from
inside `src/` directly.

## Data

Bring your own OHLCV CSVs with `timestamp, open, high, low, close, volume`
columns — e.g. Binance BTC/USDT klines — at:

```
data/BTCUSDT_15m.csv
data/BTCUSDT_1h.csv
data/BTCUSDT_4h.csv
```

Data files are not committed to this repo. Pass different paths with
`--csv_15m`, `--csv_1h`, `--csv_4h` (or the `CSV_15M` / `CSV_1H` / `CSV_4H`
environment variables used by `run.sh`).

## Check for leakage first

Before training, verify the timestamp convention (bar open vs close times)
and that no feature uses future data:

```bash
python src/check_leakage.py \
  --csv_15m data/BTCUSDT_15m.csv \
  --csv_1h data/BTCUSDT_1h.csv \
  --csv_4h data/BTCUSDT_4h.csv
```

This runs a causality test: features computed on data truncated at time T
must match the features the full dataset gives for the same rows. It exits
non-zero on any mismatch.

## Train

```bash
python src/train.py \
  --csv_15m data/BTCUSDT_15m.csv \
  --csv_1h data/BTCUSDT_1h.csv \
  --csv_4h data/BTCUSDT_4h.csv \
  --out runs/tft
```

Every setting in `config.py` is also a CLI flag (booleans as `--flag` /
`--no-flag`), e.g. `--seq_len 192 --no-htf --norm global`. Defaults:

| Setting | Default | Notes |
| --- | --- | --- |
| Input window | 96 x 15m bars | 24h of context |
| Label horizon | 32 x 15m bars | 8h triple-barrier outcome |
| Barrier width | 1.0 x vol x sqrt(horizon) | vol over 96 bars |
| Split | 70% train / 15% val / 15% test | |
| Model | d_model 32, 4 heads, dropout 0.3 | |
| Training | 40 epochs, batch 512, AdamW lr 5e-4, wd 0.05, patience 8 | class-weighted cross-entropy, bf16 autocast on CUDA |
| Normalisation | per-window z-score | `--norm global` switches to train-set statistics |
| Features | TA indicators + 1h/4h features | `--no-ta` / `--no-htf` ablate them |

Checkpoints are saved atomically, so a job killed mid-write cannot corrupt
the existing checkpoint; resume a run with `--resume` (see `run.sh resume`).

## Signals and backtest

`signals.py` turns per-bar probabilities `[p_neutral, p_long, p_short]` into
actions:

- Flat -> `LONG_ENTRY` / `SHORT_ENTRY` when that side's probability reaches
  the entry threshold (default 0.50) and beats the other side.
- Holding -> `EXIT` when conviction in the held side falls below the exit
  threshold (default 0.35), the opposite side reaches the entry threshold,
  or the max hold (default 64 bars = 16h) is reached.

`backtest.py` fills a signal at the **open of the next bar** and charges
0.04% fee and 0.01% slippage per side by default, reporting strategy
return vs buy-and-hold, annualised Sharpe, max drawdown, trade count,
win rate, and average trade.

## Run on Slurm

`run.sh` wraps the full lifecycle on a Slurm cluster (default partition
`gpu`, 1 GPU — confirm the partition name and GPU type, e.g. `gpu:a100:1`,
against your cluster docs, and check `module avail` for the environment
modules, which are commented out in the scripts):

```bash
./run.sh setup            # create venv, install dependencies (login node)
./run.sh check            # leakage + timestamp checks, before training
./run.sh test             # quick 2-epoch interactive GPU smoke test
./run.sh submit           # submit the full training job with sbatch
./run.sh resume runs/tft_<jobid>   # continue an interrupted run
./run.sh status | logs | cancel
```

Extra training flags pass through `TRAIN_ARGS`, e.g.
`TRAIN_ARGS="--no-htf --norm global" ./run.sh submit`. Outputs land in
`runs/tft_<jobid>/`; logs in `logs/`.

## Model notes

The TFT here follows the paper's structure in compact form: per-variable
embeddings with learned variable-selection weights, an LSTM sequence encoder
(no sinusoidal positional encoding — the LSTM carries order, and calendar
features such as hour / day-of-week are separate inputs), gated residual
networks, and causal multi-head self-attention over the encoded sequence,
ending in a 3-class head. `train.py` also reports mean variable-selection
weights, so you can see which inputs the model actually leans on.
