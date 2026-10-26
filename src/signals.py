"""Turn class probabilities into executable signals with a one-position-at-a-time state machine."""
import numpy as np

HOLD_SIG, LONG_ENTRY, SHORT_ENTRY, EXIT = 0, 1, 2, 3
SIG_NAMES = {0: "HOLD", 1: "LONG_ENTRY", 2: "SHORT_ENTRY", 3: "EXIT"}


def generate_signals(probs, enter_th=0.5, exit_th=0.35, max_hold=64):
    """probs: (N, 3) = [p_neutral, p_long, p_short] at each bar close.

    Returns (signals, position_after_signal) with position in {-1, 0, +1}.
      flat     -> LONG_ENTRY / SHORT_ENTRY when that side's probability >= enter_th (and beats the other)
      holding  -> EXIT when conviction fades (< exit_th), the opposite side reaches enter_th,
                  or max_hold bars have passed
    EXIT is only emitted while holding; entries only while flat.
    """
    n = len(probs)
    sig = np.zeros(n, dtype=np.int64)
    pos = np.zeros(n, dtype=np.int64)
    cur, held = 0, 0
    for i in range(n):
        _, pl, ps = probs[i]
        if cur == 0:
            if pl >= enter_th and pl > ps:
                sig[i], cur, held = LONG_ENTRY, 1, 0
            elif ps >= enter_th and ps > pl:
                sig[i], cur, held = SHORT_ENTRY, -1, 0
        else:
            held += 1
            p_own, p_opp = (pl, ps) if cur == 1 else (ps, pl)
            if p_own < exit_th or p_opp >= enter_th or held >= max_hold:
                sig[i], cur = EXIT, 0
        pos[i] = cur
    return sig, pos
