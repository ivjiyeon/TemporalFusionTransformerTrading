"""Temporal Fusion Transformer (compact, pure PyTorch).

Positional information: there is deliberately NO sinusoidal positional encoding. In the TFT the
LSTM sequence encoder runs before self-attention and encodes order by construction (the TFT paper
uses it as the replacement for positional encoding). Calendar features (hour / day-of-week) are
separate model inputs.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """Gated Residual Network."""

    def __init__(self, d_in, d_hid, d_out, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_hid)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(d_hid, 2 * d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        h = self.fc2(F.elu(self.fc1(x)))
        h = F.glu(self.gate(self.drop(h)), dim=-1)
        return self.norm(h + self.skip(x))


class GateAddNorm(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(d, 2 * d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, skip):
        return self.norm(F.glu(self.gate(self.drop(x)), dim=-1) + skip)


class VariableSelection(nn.Module):
    """Per-variable embedding + per-variable GRN, mixed by learned softmax weights.
    Vectorised over variables (einsum) so 50+ inputs stay fast."""

    def __init__(self, n_vars, d, dropout=0.1):
        super().__init__()
        self.n_vars, self.d = n_vars, d
        self.emb_w = nn.Parameter(torch.empty(n_vars, d).uniform_(-1, 1))
        self.emb_b = nn.Parameter(torch.empty(n_vars, d).uniform_(-1, 1))
        bound = 1.0 / math.sqrt(d)

        def p(*shape):
            return nn.Parameter(torch.empty(*shape).uniform_(-bound, bound))

        self.w1, self.b1 = p(n_vars, d, d), p(n_vars, d)
        self.w2, self.b2 = p(n_vars, d, d), p(n_vars, d)
        self.wg, self.bg = p(n_vars, d, 2 * d), p(n_vars, 2 * d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.weight_grn = GRN(n_vars * d, d, n_vars, dropout)

    def forward(self, x):                                            # x: (B, T, V)
        e = x.unsqueeze(-1) * self.emb_w + self.emb_b                # (B, T, V, d)
        h = F.elu(torch.einsum("btvd,vde->btve", e, self.w1) + self.b1)
        h = torch.einsum("btvd,vde->btve", h, self.w2) + self.b2
        g = F.glu(torch.einsum("btvd,vde->btve", self.drop(h), self.wg) + self.bg, dim=-1)
        proc = self.norm(g + e)                                      # (B, T, V, d)
        w = torch.softmax(self.weight_grn(e.flatten(-2)), dim=-1)    # (B, T, V)
        return (proc * w.unsqueeze(-1)).sum(dim=-2), w


class TFT(nn.Module):
    def __init__(self, n_vars, d_model=32, n_heads=4, dropout=0.3, n_classes=3, lstm_layers=1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by heads"
        self.vsn = VariableSelection(n_vars, d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.gan1 = GateAddNorm(d_model, dropout)
        self.enrich = GRN(d_model, d_model, d_model, dropout)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gan2 = GateAddNorm(d_model, dropout)
        self.ff = GRN(d_model, d_model, d_model, dropout)
        self.gan3 = GateAddNorm(d_model, dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):                                            # x: (B, T, V)
        e, w = self.vsn(x)
        h, _ = self.lstm(e)                                          # <- carries order information
        h = self.gan1(h, e)
        z = self.enrich(h)
        T = z.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)   # causal
        a, _ = self.attn(z, z, z, attn_mask=mask, need_weights=False)
        a = self.gan2(a, z)
        o = self.gan3(self.ff(a), h)
        return self.head(o[:, -1]), w
