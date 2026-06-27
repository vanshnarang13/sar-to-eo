"""Inference entry point"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMG_EXTS = (".png", ".PNG")


def load_generator(weights_path: str, device: torch.device, in_ch: int):
    """Rebuild the trained generator from the checkpoint's own config (arch-aware)."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("G", ckpt)                      # accept raw state_dict too
    m = ckpt.get("cfg", {}).get("model", {})
    arch = m.get("arch", "hybrid")
    if arch == "hybrid":
        from src.models.hybrid_cgan import HybridGenerator
        norm = nn.BatchNorm2d if m.get("norm") == "batch" else nn.InstanceNorm2d
        classes = m.get("classes", ["agri", "barrenland", "grassland", "urban"])
        G = HybridGenerator(in_ch=in_ch, out_ch=3, ngf=m.get("ngf", 64), num_classes=len(classes),
                            n_res=m.get("n_res", 9), vit_depth=m.get("vit_depth", 12),
                            vit_dim=m.get("vit_dim", 384), vit_heads=m.get("vit_heads", 6),
                            norm=norm, attn=m.get("attn", "vit"),
                            swin_windows=m.get("swin_windows", [4]))
    else:
        from src.models.hvt_cgan import HVTGenerator
        norm = nn.InstanceNorm2d if m.get("norm") == "instance" else nn.BatchNorm2d
        G = HVTGenerator(in_ch=in_ch, out_ch=3, ngf=m.get("ngf", 64),
                         n_res=m.get("n_res_blocks", 6), n_vit=m.get("n_vit_blocks", 6),
                         num_heads=m.get("num_heads", 4), sr_ratio=m.get("sr_ratio", 2), norm=norm)
    G.load_state_dict(state)
    return G.to(device).eval()


@torch.no_grad()
def translate(G, sar_uint8: np.ndarray, device, in_ch: int) -> np.ndarray:
    """SAR uint8 [0,255] HxW -> RGB uint8 [0,255] HxWx3."""
    x = sar_uint8.astype(np.float32) / 127.5 - 1.0            # [-1,1]
    t = torch.from_numpy(x)[None, None]                       # 1,1,H,W
    if in_ch > 1:                                             # adapt single VV -> trained #channels
        t = t.repeat(1, in_ch, 1, 1)
    out = G(t.to(device))
    out = out[0] if isinstance(out, tuple) else out           # Hybrid cGAN returns (img, cls_logits)
    fake = out.clamp(-1, 1)[0].cpu().numpy()                  # 3,H,W
    rgb = ((fake.transpose(1, 2, 0) + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--in_ch", type=int, default=1, help="input channels the model expects (VV=1)")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (auto if omitted)")
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    G = load_generator(args.weights, device, args.in_ch)

    files = sorted(p for p in in_dir.iterdir() if p.suffix in IMG_EXTS)
    if not files:
        raise SystemExit(f"No PNG files found in {in_dir}")
    print(f"device={device}  weights={args.weights}  files={len(files)}")
    for p in files:
        sar = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        if sar.shape != (256, 256):
            sar = np.asarray(Image.fromarray(sar).resize((256, 256), Image.BILINEAR), dtype=np.uint8)
        rgb = translate(G, sar, device, args.in_ch)
        Image.fromarray(rgb).save(out_dir / p.name)   # uint8 HxWx3 -> RGB, same filename
    print(f"wrote {len(files)} RGB images -> {out_dir}")


if __name__ == "__main__":
    main()
