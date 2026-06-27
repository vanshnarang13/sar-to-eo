
from __future__ import annotations

import torch
import torch.nn as nn


def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """B,H,W,C -> (num_windows*B), ws, ws, C."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """(num_windows*B), ws, ws, C -> B,H,W,C."""
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """MSA within a single ws x ws window + relative-position bias (Liu 2021, eq. 4)."""

    def __init__(self, dim: int, window_size: int, heads: int):
        super().__init__()
        self.ws = window_size
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        # relative position bias table: (2*ws-1)^2 x heads, indexed by intra-window rel. coords
        self.rel_bias = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))   # 2,ws,ws
        coords_flat = torch.flatten(coords, 1)                                       # 2, N
        rel = (coords_flat[:, :, None] - coords_flat[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_index", rel.sum(-1))                               # N,N

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        Bn, N, C = x.shape                                          # Bn = num_windows*B, N = ws*ws
        qkv = self.qkv(x).reshape(Bn, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                            # Bn, heads, N, head_dim
        attn = (q * self.scale) @ k.transpose(-2, -1)              # Bn, heads, N, N

        bias = self.rel_bias[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:                                       # SW-MSA: block cross-region attn
            nw = mask.shape[0]
            attn = attn.view(Bn // nw, nw, self.heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, N, N)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    """One Swin block over a grid x grid token map (no class token — Swin has none; Kong 2022).
    LN -> (shifted) window partition -> W-MSA/SW-MSA -> reverse -> residual; LN -> MLP -> residual.
    """

    def __init__(self, dim: int = 384, heads: int = 6, window_size: int = 4, shift: int = 0,
                 grid: int = 16, mlp_ratio: int = 4):
        super().__init__()
        self.grid = grid
        if window_size >= grid:                                    # window covers whole grid -> no shift
            window_size, shift = grid, 0
        self.ws = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.ws, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))
        self.register_buffer("attn_mask", self._build_mask() if self.shift > 0 else None,
                             persistent=False)

    def _build_mask(self) -> torch.Tensor:
        H = W = self.grid
        ws, shift = self.ws, self.shift
        img_mask = torch.zeros((1, H, W, 1))
        slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for h in slices:
            for w in slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mw = window_partition(img_mask, ws).view(-1, ws * ws)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape                                          # N = grid*grid
        H = W = self.grid
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        win = window_partition(x, self.ws).view(-1, self.ws * self.ws, C)
        win = self.attn(win, mask=self.attn_mask)
        x = window_reverse(win.view(-1, self.ws, self.ws, C), self.ws, H, W)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = shortcut + x.view(B, N, C)
        return x + self.mlp(self.norm2(x))
