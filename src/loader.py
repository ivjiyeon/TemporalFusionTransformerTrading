"""Torch batching.

The whole (n, F) feature matrix lives on the GPU (~40 MB for 184k x 56 floats). Each batch is
built by gathering windows with an index tensor and, in "window" mode, z-scoring every feature
inside its own window on the GPU. No DataLoader workers needed.
"""
import torch


def window_zscore(x: torch.Tensor, n_norm: int, clip: float = 5.0) -> torch.Tensor:
    """x: (B, L, F). Z-score the first n_norm features over the time axis of each window.
    The remaining columns (calendar features) are left as they are: they are already bounded, and
    a near-constant column inside a window (day-of-week) would be blown up by z-scoring."""
    a = x[..., :n_norm].float()
    mu = a.mean(dim=1, keepdim=True)
    sd = a.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    a = ((a - mu) / sd).clamp(-clip, clip)
    return torch.cat([a, x[..., n_norm:].float()], dim=-1)


class WindowBatcher:
    def __init__(self, X, y, idx, seq_len, batch_size, shuffle, n_norm, window_norm):
        self.X, self.y = X, y                                   # device tensors
        self.idx = torch.as_tensor(idx, dtype=torch.long, device=X.device)
        self.offs = torch.arange(-seq_len + 1, 1, device=X.device)
        self.bs, self.shuffle = batch_size, shuffle
        self.n_norm, self.window_norm = n_norm, window_norm

    def __len__(self):
        n = len(self.idx)
        return n // self.bs if self.shuffle else -(-n // self.bs)

    def __iter__(self):
        n = len(self.idx)
        order = torch.randperm(n, device=self.idx.device) if self.shuffle else torch.arange(n, device=self.idx.device)
        for s in range(0, n, self.bs):
            sel = self.idx[order[s:s + self.bs]]
            if self.shuffle and len(sel) < self.bs:             # drop last partial batch when training
                break
            xb = self.X[sel[:, None] + self.offs]               # (B, L, F)
            if self.window_norm:
                xb = window_zscore(xb, self.n_norm)
            yield xb, self.y[sel]
