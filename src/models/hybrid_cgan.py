"""Hybrid cGAN — Wang, Ma, Zhang (2022a), "Hybrid cGAN: Coupling Global and Local Features
for SAR-to-Optical Image Translation", IEEE TGRS 60, 5236016.

Replicated from the paper text (Sec. III). Some wiring is defined by figures (Fig. 3-7) that are
not machine-readable; those reconstructions are flagged `# FIG-RECON` and are the faithful
best-effort interpretation — verify against the paper figures if exact parity matters.

Generator (Sec. III-A): two parallel branches with bidirectional transmission.
  CNN branch  : 7x7 s1 -> two 3x3 s2 encode (256x256x1 -> 64x64x256), then 9 improved residual
                blocks (modified Res2Net + SE), then two deconv + 7x7-Tanh decode. InstanceNorm +
                LeakyReLU on all conv/deconv except the last.
  ViT branch  : 12 ViT blocks, embed dim 384, 6 heads, NO positional embedding (CNN already encodes
                position). Emits a class token -> FC -> classification logits.
  DS transmission (CNN->ViT): the 3 encoder feature maps -> 1x1 conv to 384 -> avgpool to 16x16 ->
                256x384 tokens (+LN+GELU). First feeds the ViT input (with a class token); the other
                two are added to early ViT block outputs.
  US transmission (ViT->CNN): ViT blocks 4..12 (9 of them) -> strip class token -> 16x16x384 ->
                1x1 conv to 256 -> IN+ReLU -> upsample to 64x64 -> fused into the 9 residual blocks.

Discriminator (Sec. III-B): multiscale PatchGAN at full/half/quarter resolution (RF 70/140/280),
  five conv layers, spectral norm on all conv except the first and last.

Losses live in src/losses.py (adv per-scale + L1 + VGG perceptual + classification).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# --------------------------------------------------------------------------- #
#  SE block + improved (Res2Net-style) residual block
# --------------------------------------------------------------------------- #
class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1), nn.ReLU(True),
            nn.Conv2d(ch // reduction, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ImprovedResBlock(nn.Module):
    """Res2Net-style hierarchical block (4 subsets) + SE, fused with a ViT global feature.

    paper: "FRESi-1 merged with its ViT feature map, then split evenly on channel into 4 subsets;
    each subset -> 3x3 conv; output of subset i-1 (after conv) merged into subset i before conv;
    concat -> 1x1 conv to fuse; SE before the residual connection."
    """

    def __init__(self, ch: int = 256, norm=nn.InstanceNorm2d):
        super().__init__()
        self.sub = ch // 4
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(self.sub, self.sub, 3, 1, 1),
                           norm(self.sub), nn.LeakyReLU(0.2, True)) for _ in range(4)]
        )
        self.fuse = nn.Sequential(nn.Conv2d(ch, ch, 1), norm(ch), nn.LeakyReLU(0.2, True))
        self.se = SEBlock(ch)

    def forward(self, x, vit_feat=None):
        m = x + vit_feat if vit_feat is not None else x        # FIG-RECON: merge = element-wise add
        s = [c.contiguous() for c in torch.split(m, self.sub, dim=1)]   # 4 subsets of ch/4
        y0 = self.convs[0](s[0])
        y1 = self.convs[1](s[1] + y0)                           # hierarchical (Res2Net)
        y2 = self.convs[2](s[2] + y1)
        y3 = self.convs[3](s[3] + y2)
        out = self.fuse(torch.cat([y0, y1, y2, y3], dim=1))
        return x + self.se(out)                                  # SE then residual


# --------------------------------------------------------------------------- #
#  ViT branch
# --------------------------------------------------------------------------- #
class ViTBlock(nn.Module):
    def __init__(self, dim: int = 384, heads: int = 6, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DSTrans(nn.Module):
    """CNN feature map -> ViT tokens: 1x1 conv to dim, avgpool to gridxgrid, flatten, LN+GELU."""

    def __init__(self, in_ch: int, dim: int = 384, grid: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, 1)
        self.grid = grid
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = F.adaptive_avg_pool2d(x, self.grid)                 # B,dim,16,16
        x = x.flatten(2).transpose(1, 2)                        # B,256,dim
        return self.act(self.norm(x))


class USTrans(nn.Module):
    """ViT tokens -> CNN feature map: reshape to gridxgrid, 1x1 conv to ch, IN+ReLU, upsample to hw."""

    def __init__(self, dim: int = 384, out_ch: int = 256, grid: int = 16, hw: int = 64,
                 norm=nn.InstanceNorm2d):
        super().__init__()
        self.grid, self.hw = grid, hw
        self.proj = nn.Sequential(nn.Conv2d(dim, out_ch, 1), norm(out_ch), nn.ReLU(True))

    def forward(self, tokens):                                   # tokens: B,256,dim (no class token)
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).contiguous().view(B, C, self.grid, self.grid)
        x = self.proj(x)
        return F.interpolate(x, size=(self.hw, self.hw), mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- #
#  Generator
# --------------------------------------------------------------------------- #
class HybridGenerator(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 3, ngf: int = 64, num_classes: int = 4,
                 n_res: int = 9, vit_depth: int = 12, vit_dim: int = 384, vit_heads: int = 6,
                 norm=nn.InstanceNorm2d, attn: str = "vit", swin_windows=(4,), grid: int = 16):
        super().__init__()
        self.n_res = n_res
        self.attn = attn                                        # "vit" (global MHSA) | "swin" (windowed)
        self.swin = (attn == "swin")
        dim = ngf * 4                                            # 256

        # CNN encoder (3 conv): 7x7 s1, then two 3x3 s2
        self.enc0 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7, 1, 0),
                                  norm(ngf), nn.LeakyReLU(0.2, True))
        self.enc1 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, 3, 2, 1), norm(ngf * 2),
                                  nn.LeakyReLU(0.2, True))
        self.enc2 = nn.Sequential(nn.Conv2d(ngf * 2, dim, 3, 2, 1), norm(dim),
                                  nn.LeakyReLU(0.2, True))

        # DS transmission for the 3 encoder maps
        self.ds = nn.ModuleList([DSTrans(ngf, vit_dim, grid), DSTrans(ngf * 2, vit_dim, grid),
                                 DSTrans(dim, vit_dim, grid)])

        # Transformer branch — global ViT (Wang 2022a) or windowed Swin (ablation, Liu 2021/Kong 2022).
        # Swin has no class token (Kong 2022); classification uses global-average-pooled final tokens.
        if self.swin:
            from src.models.swin_blocks import SwinBlock
            win = list(swin_windows)
            self.vit = nn.ModuleList([
                SwinBlock(vit_dim, vit_heads, window_size=win[i % len(win)],
                          shift=(win[i % len(win)] // 2 if i % 2 == 1 else 0), grid=grid)
                for i in range(vit_depth)])
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.vit = nn.ModuleList([ViTBlock(vit_dim, vit_heads) for _ in range(vit_depth)])
        self.classifier = nn.Linear(vit_dim, num_classes)

        # US transmission for ViT blocks 4..12 (n_res of them) -> CNN residual blocks
        self.us = nn.ModuleList([USTrans(vit_dim, dim) for _ in range(n_res)])
        self.res = nn.ModuleList([ImprovedResBlock(dim, norm) for _ in range(n_res)])

        # CNN decoder
        self.dec0 = nn.Sequential(nn.ConvTranspose2d(dim, ngf * 2, 3, 2, 1, output_padding=1),
                                  norm(ngf * 2), nn.LeakyReLU(0.2, True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, output_padding=1),
                                  norm(ngf), nn.LeakyReLU(0.2, True))
        self.out = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7, 1, 0), nn.Tanh())

    def forward(self, x):
        e0 = self.enc0(x)                                       # B,64,256,256
        e1 = self.enc1(e0)                                      # B,128,128,128
        e2 = self.enc2(e1)                                      # B,256,64,64  (FSAR)

        pe = [self.ds[0](e0), self.ds[1](e1), self.ds[2](e2)]   # 3 x (B,256,384)
        B = x.size(0)

        if self.swin:                                           # no class token (Swin/Kong 2022)
            t = pe[0]                                           # B,256,384
            us_feats = []
            for i, blk in enumerate(self.vit):
                t = blk(t)
                if i == 0:                                      # FIG-RECON: add pe[1],pe[2] early
                    t = t + pe[1]
                elif i == 1:
                    t = t + pe[2]
                if i >= len(self.vit) - self.n_res:             # last n_res blocks -> CNN
                    us_feats.append(self.us[len(us_feats)](t))
            cls_logits = self.classifier(t.mean(dim=1))         # global-average-pool -> classifier
        else:
            t = torch.cat([self.cls_token.expand(B, -1, -1), pe[0]], dim=1)   # B,257,384
            us_feats = []
            for i, blk in enumerate(self.vit):
                t = blk(t)
                if i == 0:                                      # FIG-RECON: add pe[1],pe[2] early
                    t = torch.cat([t[:, :1], t[:, 1:] + pe[1]], dim=1)
                elif i == 1:
                    t = torch.cat([t[:, :1], t[:, 1:] + pe[2]], dim=1)
                if i >= len(self.vit) - self.n_res:             # last n_res blocks (4..12) -> CNN
                    us_feats.append(self.us[len(us_feats)](t[:, 1:]))
            cls_logits = self.classifier(t[:, 0])               # from class token

        h = e2
        for k, block in enumerate(self.res):
            h = block(h, us_feats[k])                           # fuse ViT global feature
        out = self.out(self.dec1(self.dec0(h)))                 # B,3,256,256
        return out, cls_logits


# --------------------------------------------------------------------------- #
#  Multiscale spectral-norm PatchGAN discriminator
# --------------------------------------------------------------------------- #
class _PatchD(nn.Module):
    """Single-scale PatchGAN; spectral norm on all conv except first and last (paper Sec. III-B)."""

    def __init__(self, in_ch: int = 4, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]   # first: no SN
        nf = ndf
        for i in range(1, n_layers):
            nf_prev, nf = nf, min(ndf * 2 ** i, 512)
            layers += [spectral_norm(nn.Conv2d(nf_prev, nf, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        nf_prev, nf = nf, min(ndf * 2 ** n_layers, 512)
        layers += [spectral_norm(nn.Conv2d(nf_prev, nf, 4, 1, 1)), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(nf, 1, 4, 1, 1)]                                # last: no SN
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiScaleD(nn.Module):
    """D1/D2/D3 at full, 1/2, 1/4 resolution -> RF 70/140/280. Returns a list of logit maps."""

    def __init__(self, in_ch: int = 4, ndf: int = 64, num_D: int = 3):
        super().__init__()
        self.Ds = nn.ModuleList([_PatchD(in_ch, ndf) for _ in range(num_D)])
        self.down = nn.AvgPool2d(3, 2, 1, count_include_pad=False)

    def forward(self, sar, eo):
        x = torch.cat([sar, eo], dim=1)
        outs = []
        for i, D in enumerate(self.Ds):
            xi = x
            for _ in range(i):
                xi = self.down(xi)
            outs.append(D(xi))
        return outs                                             # list of (B,1,h,w)


def init_weights(net: nn.Module, gain: float = 0.02):
    def f(m):
        cn = m.__class__.__name__
        # skip spectral-norm-wrapped weights (parametrized) and norm/LayerNorm/Linear-cls handled below
        if hasattr(m, "weight") and m.weight is not None and ("Conv" in cn):
            if not hasattr(m, "weight_orig"):                   # not spectral-norm-parametrized
                nn.init.normal_(m.weight.data, 0.0, gain)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias.data, 0.0)
        elif "InstanceNorm" in cn and getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(f)
    return net
