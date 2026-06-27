"""Evaluation entry point matching the assignment README contract.

    python eval.py --pred_dir <generated_rgb_dir> --gt_dir <ground_truth_rgb_dir>

Computes the ranked perceptual metrics (LPIPS, FID) and the secondary pixel metrics (SSIM, PSNR)
over every PNG that exists in BOTH directories, matched by identical filename. Predictions are the
RGB images produced by infer.py; ground truth is the matching optical RGB.

Runs on a single GPU, MPS, or CPU (auto-detected), and works the same on Windows, macOS and Linux:
it loads images in a simple loop with no multiprocessing, so there is no platform-specific
DataLoader worker behaviour to worry about. FID needs a one-time InceptionV3 download; pass
--no_fid to skip it when offline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.eval import evaluate


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description="Compute LPIPS / FID / SSIM / PSNR between two dirs.")
    ap.add_argument("--pred_dir", required=True, help="dir of generated RGB PNGs (from infer.py)")
    ap.add_argument("--gt_dir", required=True, help="dir of ground-truth RGB PNGs (same filenames)")
    ap.add_argument("--no_fid", action="store_true", help="skip FID (e.g. offline / no Inception)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (auto if omitted)")
    ap.add_argument("--out", default=None, help="optional JSON path to also write the results to")
    args = ap.parse_args()

    device = _pick_device(args.device)
    res = evaluate(args.pred_dir, args.gt_dir, terrain_map=None,
                   batch_size=args.batch_size, compute_fid=not args.no_fid, device=device)

    o = res["overall"]
    print(f"\ndevice={device}  images matched={res['n_images']}")
    print("=" * 44)
    print(f"  LPIPS (down) : {o['LPIPS']:.4f}")
    if "FID" in o:
        print(f"  FID   (down) : {o['FID']:.2f}")
    print(f"  SSIM  (up)   : {o['SSIM']:.4f}")
    print(f"  PSNR  (up)   : {o['PSNR']:.3f}")
    print("=" * 44)

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
