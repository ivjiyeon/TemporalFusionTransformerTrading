"""Data pipeline: load 15m/1h/4h files -> aligned feature table -> triple-barrier labels -> splits.

No torch in this file (so it can be tested on any machine).

Look-ahead protection
  * A 1h/4h bar is attached to a 15m row only if that higher-timeframe bar has already CLOSED
    at the moment the 15m bar closes (merge_asof, direction="backward").
  * Windows never span gaps in the 15m series; labels never look across a gap.
  * Train / val labels are purged by `horizon` bars before each split boundary.
  * `check_leakage.py` verifies causality empirically.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from features import HTF_CORE, TIME, compute_features

NEUTRAL, LONG, SHORT = 0, 1, 2
TF_DELTA = {"15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}


# ----------------------------------------------------------------------------- loading
def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in ("timestamp", "open_time", "datetime", "date", "time") if c in df.columns),
                df.columns[0])
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        x = float(t.iloc[0])
        unit = "s" if x < 1e11 else "ms" if x < 1e14 else "us" if x < 1e17 else "ns"
        ts = pd.to_datetime(t, unit=unit)
    else:
        ts = pd.to_datetime(t, utc=True).dt.tz_localize(None)
    df["timestamp"] = ts.astype("datetime64[ns]")
    need = ["open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing} (found {list(df.columns)})")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df[["timestamp"] + need].reset_index(drop=True)


def close_time(ts: pd.Series, tf: str, stamp: str) -> pd.Series:
    """Moment at which a bar is complete. `stamp` says whether the file stores open or close times."""
    return ts + TF_DELTA[tf] if stamp == "open" else ts


def detect_stamp(fine: pd.DataFrame, coarse: pd.DataFrame) -> str:
    """Infer whether timestamps are bar-open or bar-close times by comparing two timeframes.
    open-stamped:  coarse.open == fine.open  at the same timestamp
    close-stamped: coarse.close == fine.close at the same timestamp"""
    m = coarse.merge(fine, on="timestamp", suffixes=("_c", "_f"))
    if len(m) < 50:
        return "unknown"
    o = float(np.isclose(m.open_c, m.open_f, rtol=1e-9).mean())
    c = float(np.isclose(m.close_c, m.close_f, rtol=1e-9).mean())
    if o > 0.9 and c < 0.5:
        return "open"
    if c > 0.9 and o < 0.5:
        return "close"
    return "unknown"


# ----------------------------------------------------------------------------- alignment
def attach_htf(base_close: pd.Series, hdf: pd.DataFrame, hfeats: pd.DataFrame, tf: str, stamp: str,
               prefix: str) -> pd.DataFrame:
    """Attach the latest COMPLETED higher-timeframe feature row to every base row."""
    right = hfeats.copy()
    right.columns = [prefix + c for c in right.columns]
    right["_avail"] = close_time(hdf.timestamp, tf, stamp).values
    right = right.sort_values("_avail")
    left = pd.DataFrame({"_key": base_close.values})
    m = pd.merge_asof(left, right, left_on="_key", right_on="_avail", direction="backward",
                      tolerance=TF_DELTA[tf])          # stale (gap) -> NaN -> window dropped
    return m.drop(columns=["_key", "_avail"])


def build_feature_table(base, h1, h4, cfg):
    """Returns (table, base_close_time, n_norm). The first n_norm columns are window-normalised;
    the remaining columns (calendar features) are passed through untouched."""
    ctime = close_time(base.timestamp, "15m", cfg.stamp).reset_index(drop=True)
    parts = [compute_features(base, cfg.ta)]
    if cfg.htf:
        for tf, hdf in (("1h", h1), ("4h", h4)):
            hf = compute_features(hdf, cfg.ta)
            hf = hf[[c for c in HTF_CORE if c in hf.columns]]
            parts.append(attach_htf(ctime, hdf, hf, tf, cfg.stamp, f"h{tf[:-1]}_"))
    price = pd.concat(parts, axis=1)

    hr = ctime.dt.hour + ctime.dt.minute / 60.0
    dow = ctime.dt.dayofweek
    tdf = pd.DataFrame({"hour_sin": np.sin(2 * np.pi * hr / 24), "hour_cos": np.cos(2 * np.pi * hr / 24),
                        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7)})[TIME]
    table = pd.concat([price, tdf], axis=1)
    return table, ctime, price.shape[1]


# ----------------------------------------------------------------------------- labels
def triple_barrier(close, vol, horizon, mult, fut_gap):
    """LONG if +barrier is hit first, SHORT if -barrier first, NEUTRAL if neither within `horizon`
    bars. barrier = mult * vol * sqrt(horizon) in log-return terms. -1 = unlabeled."""
    n = len(close)
    y = -np.ones(n, dtype=np.int64)
    m = n - horizon
    lc = np.log(close)
    rel = sliding_window_view(lc[1:], horizon)[:m] - lc[:m, None]     # row i: path of bars i+1..i+horizon
    b = (mult * vol[:m] * np.sqrt(horizon))[:, None]
    big = horizon + 1
    with np.errstate(invalid="ignore"):
        up, dn = rel >= b, rel <= -b
    iu = np.where(up.any(1), up.argmax(1), big)
    idn = np.where(dn.any(1), dn.argmax(1), big)
    lab = np.where(iu == idn, NEUTRAL, np.where(iu < idn, LONG, SHORT))
    ok = np.isfinite(vol[:m]) & (vol[:m] > 0) & ~fut_gap[:m]
    y[:m] = np.where(ok, lab, -1)
    return y


# ----------------------------------------------------------------------------- assemble
@dataclass
class Prepared:
    X: np.ndarray            # (n, F) float32
    y: np.ndarray            # (n,) int64, -1 = no label
    names: list
    n_norm: int
    open: np.ndarray
    close: np.ndarray
    ts: np.ndarray           # decision time = close of the 15m bar
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    stats: dict


def _build_prepared(cfg, log=print) -> Prepared:
    base, h1, h4 = (load_ohlcv(p) for p in (cfg.csv_15m, cfg.csv_1h, cfg.csv_4h))
    log(f"rows: 15m={len(base):,}  1h={len(h1):,}  4h={len(h4):,}   "
        f"15m range {base.timestamp.iloc[0]} -> {base.timestamp.iloc[-1]}")

    # timestamp-convention sanity check (a wrong 'open' assumption would leak future bars)
    for name, fine, coarse in (("15m vs 1h", base, h1), ("1h vs 4h", h1, h4)):
        det = detect_stamp(fine, coarse)
        log(f"timestamp convention ({name}): detected '{det}', configured '{cfg.stamp}'")
        if det == "open" and cfg.stamp == "close":
            raise ValueError("Files look OPEN-stamped but --stamp close was given: this would leak the future.")
        if det not in ("unknown", cfg.stamp):
            log("  WARNING: detected convention differs from --stamp")

    table, ctime, n_norm = build_feature_table(base, h1, h4, cfg)
    n, H, L = len(base), cfg.horizon, cfg.seq_len
    close = base.close.values.astype(float)
    vol = np.log(base.close).diff().rolling(cfg.vol_window).std().values

    contig = (base.timestamp.diff() == TF_DELTA["15m"]).values           # row continues the previous bar
    gc = np.cumsum(~contig)
    fut_gap = np.zeros(n, dtype=bool)
    fut_gap[: n - H] = (gc[H:] - gc[: n - H]) > 0                         # gap inside the label horizon
    y = triple_barrier(close, vol, H, cfg.barrier_mult, fut_gap)

    X = table.values.astype(np.float64)
    valid_row = np.isfinite(X).all(1) & contig
    cs = np.cumsum(~valid_row)
    win_ok = np.zeros(n, dtype=bool)
    win_ok[L:] = (cs[L:] - cs[: n - L]) == 0                              # rows t-L+1..t all valid
    cand = np.where(win_ok)[0]
    if len(cand) < 1000:
        raise ValueError(f"only {len(cand)} valid windows - check data / warm-up / gaps")

    tr_end = cand[int(cfg.train_frac * len(cand))]
    va_end = cand[int((cfg.train_frac + cfg.val_frac) * len(cand))]

    def labelled(lo, hi):                       # purge: label at i uses bars up to i+H
        r = cand[(cand >= lo) & (cand < hi - H)]
        return r[y[r] >= 0]

    train_idx, val_idx = labelled(0, tr_end), labelled(tr_end, va_end)
    test_idx = cand[cand >= va_end]

    Xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    stats = {}
    if cfg.norm == "global":
        rows = np.where(valid_row & (np.arange(n) < tr_end))[0]
        mu, sd = X[rows, :n_norm].mean(0), X[rows, :n_norm].std(0) + 1e-9
        Xf[:, :n_norm] = np.clip((Xf[:, :n_norm] - mu) / sd, -5, 5)
        stats = {"mu": mu.tolist(), "sd": sd.tolist()}

    log(f"features: {X.shape[1]} total ({n_norm} normalised + {X.shape[1] - n_norm} calendar)   "
        f"valid windows: {len(cand):,}")
    log(f"samples  train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
    return Prepared(X=Xf.astype(np.float32), y=y, names=list(table.columns), n_norm=n_norm,
                    open=base.open.values.astype(float), close=close, ts=ctime.values,
                    train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, stats=stats)

# ----------------------------------------------------------------------------- cache
_PREP_KEYS = ("stamp", "ta", "htf", "norm", "seq_len", "horizon", "barrier_mult", "vol_window",
              "train_frac", "val_frac")
 
 
def _cache_path(cfg) -> Path:
    """Cache file name = hash of everything that influences the preprocessed arrays."""
    settings = {k: getattr(cfg, k) for k in _PREP_KEYS}
    files = {}
    for k in ("csv_15m", "csv_1h", "csv_4h"):
        p = Path(getattr(cfg, k))
        st = p.stat()
        files[k] = [p.name, st.st_size, st.st_mtime_ns]
    here = Path(__file__).parent
    code = [hashlib.sha256((here / f).read_bytes()).hexdigest() for f in ("data.py", "features.py")]
    h = hashlib.sha256(json.dumps([settings, files, code], sort_keys=True).encode()).hexdigest()[:12]
    return Path(cfg.cache_dir) / f"prep_{h}.npz"
 
 
def _save_cache(path: Path, prep: Prepared):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({"names": prep.names, "n_norm": prep.n_norm, "stats": prep.stats})
    tmp = path.with_suffix(".tmp")                       # write-then-rename: never leaves a half-written cache
    with open(tmp, "wb") as fh:
        np.savez(fh, X=prep.X, y=prep.y, open=prep.open, close=prep.close, ts=prep.ts,
                 train_idx=prep.train_idx, val_idx=prep.val_idx, test_idx=prep.test_idx,
                 meta=np.array(meta))
    os.replace(tmp, path)
 
 
def _load_cache(path: Path) -> Prepared:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return Prepared(X=z["X"], y=z["y"], names=meta["names"], n_norm=meta["n_norm"], open=z["open"],
                        close=z["close"], ts=z["ts"], train_idx=z["train_idx"], val_idx=z["val_idx"],
                        test_idx=z["test_idx"], stats=meta["stats"])
 
 
def prepare_data(cfg, log=print) -> Prepared:
    """Load the preprocessed arrays from the cache if available, otherwise build and save them."""
    path = _cache_path(cfg) if cfg.cache else None
    if path is not None and path.exists() and not cfg.rebuild_cache:
        try:
            prep = _load_cache(path)
            log(f"loaded preprocessed data from {path}  ({path.stat().st_size / 1e6:.0f} MB)")
            log(f"features: {len(prep.names)} total ({prep.n_norm} normalised)   "
                f"samples  train={len(prep.train_idx):,}  val={len(prep.val_idx):,}  test={len(prep.test_idx):,}")
            return prep
        except Exception as e:                           # corrupt / incompatible file -> rebuild
            log(f"cache {path} unreadable ({e!r}); rebuilding")
    prep = _build_prepared(cfg, log)
    if path is not None:
        _save_cache(path, prep)
        log(f"saved preprocessed data to {path}  ({path.stat().st_size / 1e6:.0f} MB)")
    return prep