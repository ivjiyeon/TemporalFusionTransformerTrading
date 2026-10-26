"""Central configuration.

Every field below is also a command-line flag, e.g.
    python train.py --seq_len 192 --no-htf --norm global
Booleans use --flag / --no-flag.
"""
import argparse
from dataclasses import dataclass, fields


@dataclass
class Config:
    # ---- data files (OHLCV csv: timestamp/date, open, high, low, close, volume) ----
    csv_15m: str = "data/BTCUSDT_15m.csv"
    csv_1h: str = "data/BTCUSDT_1h.csv"
    csv_4h: str = "data/BTCUSDT_4h.csv"
    stamp: str = "open"          # are timestamps bar OPEN times or CLOSE times? (check_leakage.py detects it)
    out: str = "runs/tft"
    cache: bool = True           # reuse preprocessed data from cache_dir when available (--no-cache to disable)
    rebuild_cache: bool = False  # ignore an existing cache and rebuild it
    cache_dir: str = "data"

    # ---- features ----
    ta: bool = True              # add TA-library indicators (ablation: --no-ta)
    htf: bool = True             # add 1h and 4h features (ablation: --no-htf)
    norm: str = "window"         # "window" = per-window z-score (default), "global" = train-set statistics

    # ---- windows / labels (base bar = 15 minutes) ----
    seq_len: int = 96            # input window: 96 x 15m = 24h
    horizon: int = 32            # label look-ahead: 32 x 15m = 8h
    barrier_mult: float = 1.0    # triple-barrier width = mult * vol * sqrt(horizon)
    vol_window: int = 96         # bars used to estimate volatility for the barrier
    train_frac: float = 0.70
    val_frac: float = 0.15       # test = remainder

    # ---- model / optimisation ----
    d_model: int = 32
    heads: int = 4
    dropout: float = 0.3
    epochs: int = 40
    batch: int = 512
    lr: float = 5e-4
    weight_decay: float = 0.05
    patience: int = 8
    seed: int = 42
    resume: bool = False

    # ---- signal rules + backtest ----
    enter_th: float = 0.50       # min class probability to open a position
    exit_th: float = 0.35        # exit when probability of the held side falls below this
    max_hold: int = 64           # force exit after this many bars (64 x 15m = 16h)
    fee: float = 0.0004          # per side
    slip: float = 0.0001         # per side
    bars_per_year: int = 35040   # 365 * 24 * 4, for annualised Sharpe


CHOICES = {"stamp": ["open", "close"], "norm": ["window", "global"]}


def parse_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(description="TFT BTC/USDT trading-signal model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for f in fields(Config):
        flag = "--" + f.name
        if isinstance(f.default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default, choices=CHOICES.get(f.name))
    return Config(**vars(ap.parse_args(argv)))
