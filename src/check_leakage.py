"""Run this before training.

1. Timestamp convention: are your files stamped with bar OPEN or bar CLOSE times?
2. Causality test: features computed on data truncated at time T must equal the features the full
   dataset gives for the same rows. If any indicator or any 1h/4h alignment used the future, the
   two versions differ and this script fails.

    python check_leakage.py --csv_15m data/BTCUSDT_15m.csv --csv_1h data/BTCUSDT_1h.csv --csv_4h data/BTCUSDT_4h.csv
"""
import sys

import numpy as np

from config import parse_args
from data import build_feature_table, close_time, detect_stamp, load_ohlcv


def main():
    cfg = parse_args()
    base, h1, h4 = (load_ohlcv(p) for p in (cfg.csv_15m, cfg.csv_1h, cfg.csv_4h))

    print("== timestamp convention ==")
    for name, fine, coarse in (("15m vs 1h", base, h1), ("1h vs 4h", h1, h4)):
        print(f"  {name}: detected '{detect_stamp(fine, coarse)}'   (configured --stamp {cfg.stamp})")

    print("== causality test ==")
    full, _, _ = build_feature_table(base, h1, h4, cfg)
    rng = np.random.default_rng(0)
    cuts = rng.integers(int(0.3 * len(base)), int(0.95 * len(base)), size=4)
    ok_all = True
    for k in cuts:
        bt = base.iloc[: k + 1]
        cut = close_time(bt.timestamp, "15m", cfg.stamp).iloc[-1]
        h1t = h1[close_time(h1.timestamp, "1h", cfg.stamp) <= cut]
        h4t = h4[close_time(h4.timestamp, "4h", cfg.stamp) <= cut]
        part, _, _ = build_feature_table(bt, h1t, h4t, cfg)
        a, b = full.iloc[: k + 1].values, part.values
        same = np.isclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)
        ok = bool(same.all())
        ok_all &= ok
        print(f"  cut at row {k:>7,} ({cut}): {'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad = np.where(~same.all(0))[0]
            print("    differing columns:", [full.columns[i] for i in bad][:10])
    print("RESULT:", "PASS - no look-ahead detected" if ok_all else "FAIL - features use future data")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
