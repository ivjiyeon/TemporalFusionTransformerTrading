"""Feature engineering for one OHLCV timeframe.

Design rules
  * Everything is causal (rolling / EWM only): row t uses data up to and including bar t.
  * Everything is scale-free (returns, ratios to close, bounded oscillators).
    The notebook's raw-dollar indicators (EMA, Bollinger bands, ...) are ~0.9999 correlated with
    price, so they add almost nothing and break under any non-window normalisation.
"""
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import ChaikinMoneyFlowIndicator, OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

# plain price / volume statistics
PRICE_VOL = ["ret1", "ret4", "ret12", "hl", "oc", "vol_s", "vol_l", "vol_z"]

# indicators from the `ta` library (same family as the notebook), converted to scale-free form
TA = ["rsi", "macd", "macd_sig", "macd_diff", "bb_pband", "bb_wband", "atr", "stoch_k", "stoch_d",
      "cmf", "obv_z", "vwap_gap", "ema_gap_20", "ema_gap_50", "ema_gap_100", "ema_gap_200"]

# subset used for the 1h and 4h context (keeps the variable count and warm-up manageable)
HTF_CORE = ["ret1", "ret4", "hl", "vol_s", "vol_z", "rsi", "macd_diff", "bb_pband", "bb_wband",
            "atr", "stoch_k", "cmf", "ema_gap_20", "ema_gap_50"]

# calendar features (bounded; NOT window-normalised)
TIME = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def compute_features(df: pd.DataFrame, use_ta: bool = True) -> pd.DataFrame:
    c, o, h, l, v = (df[k].astype(float) for k in ("close", "open", "high", "low", "volume"))
    lc = np.log(c)
    lr = lc.diff()
    lv = np.log1p(v)

    f = pd.DataFrame(index=df.index)
    f["ret1"] = lr
    f["ret4"] = lc.diff(4)
    f["ret12"] = lc.diff(12)
    f["hl"] = np.log(h / l)
    f["oc"] = np.log(c / o)
    f["vol_s"] = lr.rolling(14).std()
    f["vol_l"] = lr.rolling(50).std()
    f["vol_z"] = (lv - lv.rolling(50).mean()) / (lv.rolling(50).std() + 1e-9)

    if use_ta:
        f["rsi"] = (RSIIndicator(c, window=14).rsi() - 50) / 50
        m = MACD(c, window_slow=26, window_fast=12, window_sign=9)
        f["macd"] = m.macd() / c
        f["macd_sig"] = m.macd_signal() / c
        f["macd_diff"] = m.macd_diff() / c
        bb = BollingerBands(c, window=20, window_dev=2)
        f["bb_pband"] = bb.bollinger_pband()          # position inside the bands
        f["bb_wband"] = bb.bollinger_wband()          # band width relative to the mid line
        f["atr"] = AverageTrueRange(h, l, c, window=14).average_true_range() / c
        st = StochasticOscillator(h, l, c, window=14, smooth_window=3)
        f["stoch_k"] = (st.stoch() - 50) / 50
        f["stoch_d"] = (st.stoch_signal() - 50) / 50
        f["cmf"] = ChaikinMoneyFlowIndicator(h, l, c, v, window=20).chaikin_money_flow()
        obv = OnBalanceVolumeIndicator(c, v).on_balance_volume()   # cumulative: only its changes matter
        f["obv_z"] = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std() + 1e-9)
        vwap = VolumeWeightedAveragePrice(h, l, c, v, window=14).volume_weighted_average_price()
        f["vwap_gap"] = np.log(c / vwap)
        for w in (20, 50, 100, 200):
            f[f"ema_gap_{w}"] = np.log(c / EMAIndicator(c, window=w).ema_indicator())

    return f.replace([np.inf, -np.inf], np.nan)


def feature_group(name: str) -> str:
    """Group label used to summarise variable-selection weights."""
    if name in TIME:
        return "time"
    if name.startswith("h1_"):
        return "1h"
    if name.startswith("h4_"):
        return "4h"
    return "15m_ta" if name in TA else "15m_price_vol"
