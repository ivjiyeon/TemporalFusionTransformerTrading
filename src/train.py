"""Train the TFT classifier, then generate signals and backtest on the held-out test period.

Project layout
  config.py          all settings (also CLI flags)
  features.py        TA / price features for one timeframe
  data.py            load 15m/1h/4h, align without look-ahead, labels, splits
  loader.py          GPU window batching + per-window normalisation
  model.py           Temporal Fusion Transformer
  signals.py         probabilities -> LONG_ENTRY / SHORT_ENTRY / EXIT (one position at a time)
  backtest.py        fills at next open, fees, slippage
  train.py           this file
  check_leakage.py   causality + timestamp-convention checks (run before training)
"""
import json
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from backtest import backtest
from config import parse_args
from data import prepare_data
from features import feature_group
from loader import WindowBatcher
from model import TFT
from signals import SIG_NAMES, generate_signals


def safe_save(obj, path):
    """Atomic save: a job killed mid-write can't corrupt the existing checkpoint."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def amp(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def run_epoch(model, batcher, crit, opt=None, sched=None, clip=1.0):
    train = opt is not None
    model.train(train)
    device = next(model.parameters()).device
    tot, n, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in batcher:
            with amp(device):
                logits, _ = model(xb)
            loss = crit(logits.float(), yb)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                if sched:
                    sched.step()
            tot += loss.item() * len(yb)
            n += len(yb)
            correct += (logits.argmax(-1) == yb).sum().item()
    return tot / n, correct / n


@torch.no_grad()
def predict(model, batcher):
    """Returns class probabilities and the mean variable-selection weight of every input feature."""
    model.eval()
    device = next(model.parameters()).device
    probs, wsum, cnt = [], None, 0
    for xb, _ in batcher:
        with amp(device):
            logits, w = model(xb)
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        s = w.float().sum(dim=(0, 1))
        wsum = s if wsum is None else wsum + s
        cnt += w.shape[0] * w.shape[1]
    return np.concatenate(probs), (wsum / cnt).cpu().numpy()


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  norm={cfg.norm}  ta={cfg.ta}  htf={cfg.htf}", flush=True)

    # ---------------------------------------------------------------- data
    prep = prepare_data(cfg)
    Xt = torch.as_tensor(prep.X, device=device)
    yt = torch.as_tensor(prep.y, device=device)
    window_norm = cfg.norm == "window"
    mk = lambda idx, shuffle: WindowBatcher(Xt, yt, idx, cfg.seq_len, cfg.batch, shuffle, prep.n_norm, window_norm)
    tr_dl, va_dl = mk(prep.train_idx, True), mk(prep.val_idx, False)

    cnt = np.bincount(prep.y[prep.train_idx], minlength=3)
    print(f"train class balance (neutral/long/short): {cnt}  "
          f"| reference loss for an uninformed model = ln(3) = 1.0986", flush=True)
    br = np.diff(prep.test_idx)
    if (br != 1).any():
        print(f"WARNING: test windows are not contiguous ({(br != 1).sum()} breaks) - "
              f"backtest treats them as consecutive bars", flush=True)

    # ---------------------------------------------------------------- model
    model = TFT(len(prep.names), cfg.d_model, cfg.heads, cfg.dropout).to(device)
    print(f"variables={len(prep.names)}  params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    w = torch.tensor(cnt.sum() / (3 * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(tr_dl),
                                                pct_start=0.1)

    last_p, best_p = os.path.join(cfg.out, "last.pt"), os.path.join(cfg.out, "best.pt")
    start, best, bad = 0, float("inf"), 0
    if cfg.resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"resumed from epoch {start}", flush=True)

    # ---------------------------------------------------------------- train
    for ep in range(start, cfg.epochs):
        t0 = time.time()
        trl, tra = run_epoch(model, tr_dl, crit, opt, sched)
        val, vaa = run_epoch(model, va_dl, crit)
        print(f"ep {ep + 1:03d}  train {trl:.4f}/{tra:.3f}  val {val:.4f}/{vaa:.3f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
        if val < best:
            best, bad = val, 0
            safe_save(model.state_dict(), best_p)
        else:
            bad += 1
        safe_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                   "epoch": ep, "best": best, "bad": bad}, last_p)
        if bad >= cfg.patience:
            print("early stopping", flush=True)
            break

    # ---------------------------------------------------------------- test
    model.load_state_dict(torch.load(best_p, map_location=device))
    probs, wmean = predict(model, mk(prep.test_idx, False))
    lab = prep.y[prep.test_idx]
    m = lab >= 0
    pred = probs.argmax(1)
    maj = np.bincount(lab[m], minlength=3).max() / m.sum()
    print(f"\ntest accuracy {(pred[m] == lab[m]).mean():.3f}   (always-majority-class: {maj:.3f})")
    cm = np.zeros((3, 3), int)
    for t_, p_ in zip(lab[m], pred[m]):
        cm[t_, p_] += 1
    print("confusion (rows=true, cols=pred; neutral/long/short):\n", cm)

    # ---------------------------------------------------------------- feature usage
    imp = pd.DataFrame({"feature": prep.names, "mean_weight": wmean})
    imp["group"] = imp.feature.map(feature_group)
    imp = imp.sort_values("mean_weight", ascending=False)
    imp.to_csv(os.path.join(cfg.out, "feature_importance.csv"), index=False)
    print("\nvariable-selection weight by group (relative attention, not causal importance):")
    print(imp.groupby("group").mean_weight.sum().sort_values(ascending=False).round(3).to_string())
    print("top 10:", ", ".join(f"{f}({v:.3f})" for f, v in zip(imp.feature[:10], imp.mean_weight[:10])))

    # ---------------------------------------------------------------- signals + backtest
    sig, pos = generate_signals(probs, cfg.enter_th, cfg.exit_th, cfg.max_hold)
    stats, curve = backtest(prep.open[prep.test_idx], prep.close[prep.test_idx], sig,
                            cfg.fee, cfg.slip, cfg.bars_per_year)
    print("\nbacktest (test period only):")
    for k, v in stats.items():
        print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")

    pd.DataFrame({"timestamp": prep.ts[prep.test_idx], "p_neutral": probs[:, 0], "p_long": probs[:, 1],
                  "p_short": probs[:, 2], "signal": [SIG_NAMES[s] for s in sig], "position": pos,
                  "equity": curve}).to_csv(os.path.join(cfg.out, "test_signals.csv"), index=False)
    with open(os.path.join(cfg.out, "run_info.json"), "w") as fh:
        json.dump({"config": asdict(cfg), "stats": stats, "features": prep.names, "norm_stats": prep.stats},
                  fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"saved to {cfg.out}/")


if __name__ == "__main__":
    main()
