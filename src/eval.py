"""Compute LPIPS / FID / SSIM / PSNR between a prediction dir and a ground-truth dir.

    python -m src.eval --pred_dir <generated_rgb> --gt_dir <gt_rgb> [--terrain_map map.json]

Matches predictions to ground truth by identical filename. Optionally reports a per-terrain
breakdown if given a JSON mapping {filename: terrain} (writable from a saved split via
`scripts/export_terrain_map.py` or the dataset index).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.metrics import MetricBank


def load_rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr.transpose(2, 0, 1))  # 3,H,W in [-1,1]


def evaluate(pred_dir, gt_dir, terrain_map=None, batch_size=32, compute_fid=True, device="cpu"):
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    names = sorted(p.name for p in pred_dir.glob("*.png") if (gt_dir / p.name).exists())
    if not names:
        raise SystemExit("No matching filenames between pred_dir and gt_dir.")

    overall = MetricBank(device=device, compute_fid=compute_fid)
    per_terrain = defaultdict(lambda: MetricBank(device=device, compute_fid=False)) if terrain_map else {}

    preds, gts, terrs = [], [], []

    def flush():
        if not preds:
            return
        p = torch.stack(preds); g = torch.stack(gts)
        overall.update(p, g)
        if terrain_map:
            for t in set(terrs):
                m = [i for i, x in enumerate(terrs) if x == t]
                per_terrain[t].update(p[m], g[m])
        preds.clear(); gts.clear(); terrs.clear()

    for n in names:
        preds.append(load_rgb(pred_dir / n)); gts.append(load_rgb(gt_dir / n))
        if terrain_map:
            terrs.append(terrain_map.get(n, "unknown"))
        if len(preds) >= batch_size:
            flush()
    flush()

    result = {"n_images": len(names), "overall": overall.compute()}
    if terrain_map:
        result["per_terrain"] = {t: b.compute() for t, b in per_terrain.items()}
    return result


def evaluate_checkpoint_on_split(weights, root, split, which="test", seed=42,
                                 batch_size=32, compute_fid=True, device=None, in_ch=1,
                                 max_images=None):
    """Run a trained generator over a dataset split and report metrics (overall + per-terrain).

    Convenience path for Modal/local eval that reuses the dataset index instead of pre-dumped dirs.
    """
    import torch
    from collections import defaultdict
    from src.data.dataset import index_dataset, build_splits, SAR2EODataset
    from src.metrics import MetricBank
    from infer import load_generator

    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    samples = index_dataset(root)
    idxs = build_splits(samples, strategy=split, seed=seed)[which]
    if max_images:
        idxs = idxs[:max_images]
    ds = SAR2EODataset(samples, idxs, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4)
    G = load_generator(weights, torch.device(device), in_ch)

    overall = MetricBank(device="cpu", compute_fid=compute_fid)
    per_terrain = defaultdict(lambda: MetricBank(device="cpu", compute_fid=False))
    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            if in_ch > 1:
                sar = sar.repeat(1, in_ch, 1, 1)
            out = G(sar)
            out = out[0] if isinstance(out, tuple) else out   # Hybrid cGAN returns (img, cls_logits)
            fake = out.clamp(-1, 1).cpu()
            eo = batch["eo"]
            overall.update(fake, eo)
            for t in set(batch["terrain"]):
                m = [i for i, x in enumerate(batch["terrain"]) if x == t]
                per_terrain[t].update(fake[m], eo[m])
    return {"split": split, "which": which, "n_images": len(idxs),
            "overall": overall.compute(),
            "per_terrain": {t: b.compute() for t, b in per_terrain.items()}}


def evaluate_checkpoint_on_dir(weights, split_root, batch_size=32, compute_fid=True,
                               device=None, in_ch=1, max_images=None):
    """Evaluate a generator on a physical split directory (e.g. v_2_split/test).

    Indexes the directory the same way training indexes v_2_split/train, so train/val/test
    come from one make_split_dirs partition and are guaranteed disjoint (no leakage).
    """
    import torch
    from collections import defaultdict
    from src.data.dataset import index_dataset, SAR2EODataset
    from src.metrics import MetricBank
    from infer import load_generator

    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    samples = index_dataset(split_root)
    idxs = list(range(len(samples)))
    if max_images:
        idxs = idxs[:max_images]
    ds = SAR2EODataset(samples, idxs, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4)
    G = load_generator(weights, torch.device(device), in_ch)

    overall = MetricBank(device="cpu", compute_fid=compute_fid)
    per_terrain = defaultdict(lambda: MetricBank(device="cpu", compute_fid=False))
    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            if in_ch > 1:
                sar = sar.repeat(1, in_ch, 1, 1)
            out = G(sar)
            out = out[0] if isinstance(out, tuple) else out   # Hybrid cGAN returns (img, cls_logits)
            fake = out.clamp(-1, 1).cpu()
            eo = batch["eo"]
            overall.update(fake, eo)
            for t in set(batch["terrain"]):
                m = [i for i, x in enumerate(batch["terrain"]) if x == t]
                per_terrain[t].update(fake[m], eo[m])
    return {"split_root": str(split_root), "n_images": len(idxs),
            "overall": overall.compute(),
            "per_terrain": {t: b.compute() for t, b in per_terrain.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", help="dir of generated RGB PNGs (contract-style eval)")
    ap.add_argument("--gt_dir", help="dir of ground-truth RGB PNGs")
    ap.add_argument("--terrain_map", default=None, help="optional JSON {filename: terrain}")
    # checkpoint+split eval (convenience)
    ap.add_argument("--weights", help="checkpoint to eval directly on a dataset split")
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--split", default="random", choices=["random", "scene"])
    ap.add_argument("--which", default="test", choices=["train", "val", "test"])
    ap.add_argument("--no_fid", action="store_true")
    ap.add_argument("--out", default=None, help="optional JSON path to write results")
    args = ap.parse_args()

    if args.weights:
        res = evaluate_checkpoint_on_split(args.weights, args.root, args.split, args.which,
                                           compute_fid=not args.no_fid)
    else:
        tmap = json.loads(Path(args.terrain_map).read_text()) if args.terrain_map else None
        res = evaluate(args.pred_dir, args.gt_dir, tmap, compute_fid=not args.no_fid)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
