
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, mode: str = "vanilla", real_label: float = 1.0,
                 fake_label: float = 0.0, label_smoothing: float = 0.0):
        super().__init__()
        self.real_label = real_label - label_smoothing   # HVT: smoothing 0 -> real=1
        self.fake_label = fake_label
        if mode == "vanilla":          # original cGAN: BCE over D logits (paper eq. 8)
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":          # least-squares alternative
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"unknown gan_mode: {mode}")

    def _target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        return torch.full_like(pred, self.real_label if is_real else self.fake_label)

    def forward(self, pred, is_real: bool):
        # Single tensor (HVT, single-scale) OR a list (Hybrid cGAN multiscale D) -> summed.
        # SOURCED: Wang 2022a eq. 2 — "sum_{k=1,2,3} L_cGAN(G, D_k)", not averaged.
        if isinstance(pred, (list, tuple)):
            return sum(self.loss(p, self._target(p, is_real)) for p in pred)
        return self.loss(pred, self._target(pred, is_real))


class VGGPerceptualLoss(nn.Module):
    """VGG16 feature-space L2, used by Hybrid cGAN (Wang 2022a, eq. 4): sum_j ||phi_j(G(x))-phi_j(y)||^2
    over selected relu layers. ImageNet VGG16 is a *training-time* loss only — never imported by
    infer.py. Inputs are in [-1,1]; remapped to ImageNet-normalised [0,1] internally.

    Optimization: real (ground-truth) features are computed without grad since they don't
    participate in G's backward pass. This avoids building an autograd graph for half the VGG work.
    """

    SLICES = (3, 8, 15, 22)   # relu1_2, relu2_2, relu3_3, relu4_3

    def __init__(self):
        super().__init__()
        import torchvision
        vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x):
        x = (x + 1.0) * 0.5                       # [-1,1] -> [0,1]
        return (x - self.mean) / self.std

    def _features(self, x):
        feats = []
        last = max(self.SLICES)
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.SLICES:
                feats.append(x)
            if i >= last:
                break
        return feats

    def forward(self, fake, real):
        f = self._prep(fake)
        with torch.no_grad():
            real_feats = self._features(self._prep(real))
        fake_feats = self._features(f)
        return sum(F.mse_loss(ff, rf) for ff, rf in zip(fake_feats, real_feats))
