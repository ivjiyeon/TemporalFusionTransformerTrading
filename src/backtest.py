"""Simple event backtest. Signal at the close of bar i is filled at the OPEN of bar i+1."""
import math

import numpy as np

from signals import EXIT, LONG_ENTRY, SHORT_ENTRY


def backtest(open_, close, sig, fee=0.0004, slip=0.0001, bars_per_year=35040):
    n = len(sig)
    eq, pos, entry = 1.0, 0, 0.0
    curve = np.ones(n)
    trades = []
    for i in range(n - 1):
        px = open_[i + 1]
        s = sig[i]
        if s == EXIT:
            assert pos != 0, "EXIT without a position"
            exit_px = px * (1 - slip * pos)
            pnl = pos * (exit_px / entry - 1)
            eq *= (1 + pnl) * (1 - fee)
            trades.append(pnl - 2 * fee)
            pos = 0
        elif s in (LONG_ENTRY, SHORT_ENTRY):
            assert pos == 0, "entry while already in a position"
            pos = 1 if s == LONG_ENTRY else -1
            entry = px * (1 + slip * pos)
            eq *= 1 - fee
        curve[i + 1] = eq * (1 + pos * (close[i + 1] / entry - 1)) if pos else eq
    if pos != 0:                                   # close any open position at the end
        pnl = pos * (close[-1] / entry - 1)
        eq *= (1 + pnl) * (1 - fee)
        trades.append(pnl - 2 * fee)
        curve[-1] = eq
    rets = np.diff(curve) / curve[:-1]
    sharpe = rets.mean() / (rets.std() + 1e-12) * math.sqrt(bars_per_year)
    mdd = (curve / np.maximum.accumulate(curve) - 1).min()
    t = np.array(trades)
    stats = {
        "strategy_return": float(curve[-1] - 1),
        "buy_and_hold_return": float(close[-1] / open_[1] - 1),
        "sharpe_ann": float(sharpe),
        "max_drawdown": float(mdd),
        "n_trades": int(len(t)),
        "win_rate": float((t > 0).mean()) if len(t) else 0.0,
        "avg_trade": float(t.mean()) if len(t) else 0.0,
    }
    return stats, curve
