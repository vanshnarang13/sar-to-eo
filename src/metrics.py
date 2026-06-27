"""Evaluation metrics: LPIPS, FID (primary, ranked) + SSIM, PSNR (secondary).

All inputs are batches of images in [-1,1]. A MetricBank accumulates over batches so FID (which
needs the full distribution) and the averaged per-image metrics can be computed together.
"""
from __future__ import annotations

import lpips
import torch
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    FrechetInceptionDistance,
)


def to_01(x):
    return (x.clamp(-1, 1) + 1) / 2


def to_uint8(x):
    return (to_01(x) * 255).round().to(torch.uint8)


class MetricBank:
    """Accumulate predictions/targets, then `.compute()` LPIPS/FID/SSIM/PSNR over everything seen.

    FID needs InceptionV3 weights (one-time download). If they are unavailable (offline / no disk),
    FID is skipped gracefully rather than crashing training — it still runs on Modal where the
    download succeeds. Set `compute_fid=False` to skip it explicitly (e.g. fast local smoke tests).
    """

    def __init__(self, device="cpu", compute_fid=True):
        self.device = device
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.fid = None
        if compute_fid:
            try:
                self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            except Exception as e:  # offline / disk full -> skip FID, keep everything else
                print(f"[metrics] FID unavailable ({type(e).__name__}); skipping FID this run.")
        self._lpips_sum, self._n = 0.0, 0

    @torch.no_grad()
    def update(self, pred, target):
        pred, target = pred.to(self.device), target.to(self.device)
        self._lpips_sum += self.lpips(pred.clamp(-1, 1), target.clamp(-1, 1)).sum().item()
        self._n += pred.size(0)
        p01, t01 = to_01(pred), to_01(target)
        self.psnr.update(p01, t01)
        self.ssim.update(p01, t01)
        if self.fid is not None:
            self.fid.update(t01, real=True)     # normalize=True expects [0,1] float
            self.fid.update(p01, real=False)

    @torch.no_grad()
    def compute(self) -> dict:
        out = {
            "LPIPS": self._lpips_sum / max(self._n, 1),
            "SSIM": float(self.ssim.compute()),
            "PSNR": float(self.psnr.compute()),
        }
        if self.fid is not None:
            out["FID"] = float(self.fid.compute())
        return out

    def reset(self):
        self._lpips_sum, self._n = 0.0, 0
        self.psnr.reset(); self.ssim.reset()
        if self.fid is not None:
            self.fid.reset()
