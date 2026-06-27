"""Train the Hybrid cGAN for SAR->EO translation.

All hyperparameters and loss weights come from the config, so each ablation is one config file.

    python -m src.train --config configs/hybrid_cgan_vit.yaml
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import index_dataset, build_splits, save_splits, SAR2EODataset, TERRAINS
from src.losses import GANLoss, VGGPerceptualLoss
from src.metrics import MetricBank
from src.utils import set_seed, get_device, load_config, LossLogger, WandbRun


def _unwrap(model):
    """Return the underlying module from a DataParallel / compiled wrapper (handles nesting)."""
    m = model
    while hasattr(m, "_orig_mod") or hasattr(m, "module"):
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        elif hasattr(m, "module"):
            m = m.module
    return m


def gen_forward(G, x):
    """Generators may return an image (HVT) or (image, class_logits) (Hybrid cGAN). Normalise."""
    out = G(x)
    return out if isinstance(out, tuple) else (out, None)


def _atomic_save(obj, path):
    """Write to a temp file then os.replace -> the real checkpoint is always a complete file.
    Prevents a corrupt last.pt/best.pt if the run is interrupted (or the Modal volume is committed)
    mid-write: the previous good checkpoint stays intact until the new one is fully on disk."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)   # atomic rename on the same filesystem


def build_loaders(cfg):
    d = cfg["data"]
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    if "train_root" in d and "val_root" in d:
        # Pre-split directories: every file in each dir belongs to that split.
        # No runtime splitting — zero leakage, identical data across all ablations.
        def cap(samples, n):
            return samples[:n] if n else samples
        tr_samples = cap(index_dataset(d["train_root"]), d.get("max_train"))
        va_samples = cap(index_dataset(d["val_root"]), d.get("max_val"))
        tr = SAR2EODataset(tr_samples, list(range(len(tr_samples))), augment=d["augment"])
        va = SAR2EODataset(va_samples, list(range(len(va_samples))), augment=False)
    else:
        # Fallback: runtime split (smoke tests / backward compat only)
        samples = index_dataset(d["root"])
        splits = build_splits(samples, strategy=d["split"],
                              seed=cfg["seed"], ratios=tuple(d["ratios"]))
        save_splits(splits, samples, out / f"split_{d['split']}.json")
        def cap(idxs, n):
            return idxs[:n] if n else idxs
        tr = SAR2EODataset(samples, cap(splits["train"], d.get("max_train")), augment=d["augment"])
        va = SAR2EODataset(samples, cap(splits["val"], d.get("max_val")), augment=False)

    bs = cfg["train"]["batch_size"]
    print(f"data: train={len(tr)} val={len(va)} samples  (batch_size={bs})", flush=True)
    if len(tr) < bs:
        print(f"  WARNING: train samples ({len(tr)}) < batch_size ({bs}) with drop_last=True "
              f"-> 0 steps/epoch. Check the dataset is unpacked on the volume.", flush=True)
    nw = cfg["train"].get("num_workers", 2)
    pin = torch.cuda.is_available()
    pw  = nw > 0
    pf  = 3 if nw > 0 else None
    return (DataLoader(tr, bs, shuffle=True, num_workers=nw, drop_last=True,
                       pin_memory=pin, persistent_workers=pw, prefetch_factor=pf),
            DataLoader(va, bs, shuffle=False, num_workers=nw,
                       pin_memory=pin, persistent_workers=pw, prefetch_factor=pf))


def make_models(cfg, device):
    """Build the Hybrid cGAN generator and multiscale discriminator."""
    m = cfg["model"]
    from src.models.hybrid_cgan import HybridGenerator, MultiScaleD, init_weights
    classes = m.get("classes", list(TERRAINS))
    norm = nn.BatchNorm2d if m.get("norm") == "batch" else nn.InstanceNorm2d
    G = HybridGenerator(in_ch=1, out_ch=3, ngf=m.get("ngf", 64), num_classes=len(classes),
                        n_res=m.get("n_res", 9), vit_depth=m.get("vit_depth", 12),
                        vit_dim=m.get("vit_dim", 384), vit_heads=m.get("vit_heads", 6),
                        norm=norm, attn=m.get("attn", "vit"),
                        swin_windows=m.get("swin_windows", [4])).to(device)
    init_weights(G)
    D = MultiScaleD(in_ch=4, ndf=m.get("ndf", 64), num_D=m.get("num_D", 3)).to(device)
    init_weights(D)
    arch = m.get("arch", "hybrid")
    n_G = sum(p.numel() for p in G.parameters()) / 1e6
    n_D = sum(p.numel() for p in D.parameters()) / 1e6
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"arch={arch}  G={n_G:.1f}M  D={n_D:.1f}M  GPUs={max(n_gpus, 1)}")

    if n_gpus > 1:
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
        print(f"  DataParallel on {n_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")

    if cfg.get("compile", True) and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            G = torch.compile(G)
            D = torch.compile(D)
            print("  torch.compile enabled (first batch will be slow)")
        except Exception as e:
            print(f"  torch.compile skipped ({e})")

    return G, D


@torch.no_grad()
def validate(G, loader, device, lossw, perc, l1, max_metric_batches, compute_fid=True):
    G.eval()
    bank = MetricBank(device=device, compute_fid=compute_fid)
    agg = {"val_g_l1": 0.0, "val_g_perc": 0.0, "n": 0}
    for bi, batch in enumerate(loader):
        sar, eo = batch["sar"].to(device), batch["eo"].to(device)
        fake, _ = gen_forward(G, sar)
        agg["val_g_l1"] += l1(fake, eo).item() * sar.size(0)
        if perc is not None:
            agg["val_g_perc"] += perc(fake, eo).item() * sar.size(0)
        agg["n"] += sar.size(0)
        if max_metric_batches is None or bi < max_metric_batches:
            bank.update(fake, eo)
    n = max(agg["n"], 1)
    metrics = bank.compute()
    val = {f"val_{k}": v for k, v in metrics.items()}
    val["val_g_l1"] = agg["val_g_l1"] / n
    val["val_g_perc"] = agg["val_g_perc"] / n
    val["val_g_total"] = (lossw["lambda_l1"] * val["val_g_l1"]
                          + lossw.get("lambda_perc", 0) * val["val_g_perc"])
    return val


def _cache_vis_batch(val_loader, n_per_terrain: int = 2, device="cpu"):
    """Grab a fixed set of val samples (up to n_per_terrain per terrain) for visual logging."""
    seen: dict[str, int] = {}
    sars, eos, names = [], [], []
    for batch in val_loader:
        for i in range(batch["sar"].size(0)):
            t = batch["terrain"][i]
            if seen.get(t, 0) >= n_per_terrain:
                continue
            sars.append(batch["sar"][i])
            eos.append(batch["eo"][i])
            names.append(f"{t}/{batch['name'][i]}")
            seen[t] = seen.get(t, 0) + 1
        if all(v >= n_per_terrain for v in seen.values()) and len(seen) >= 4:
            break
    return (torch.stack(sars).to(device),
            torch.stack(eos).to(device),
            names)


def _to_pil(t: torch.Tensor):
    """Convert a (C, H, W) tensor in [-1,1] to a PIL image."""
    from PIL import Image as PILImage
    import numpy as np
    arr = ((t.clamp(-1, 1).cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)   # grayscale SAR → RGB for display
    return PILImage.fromarray(arr)


@torch.no_grad()
def _make_image_payload(G, vis_sar, vis_eo, vis_names, wb) -> dict:
    """Generate fakes from the cached vis batch and return a W&B image dict."""
    import numpy as np
    from PIL import Image as PILImage
    if wb.run is None:
        return {}
    G.eval()
    fake, _ = gen_forward(G, vis_sar)
    fake = fake.clamp(-1, 1)
    images = []
    for i, name in enumerate(vis_names):
        sar_img  = _to_pil(vis_sar[i].expand(3, -1, -1))   # repeat 1ch→3ch for display
        fake_img = _to_pil(fake[i])
        real_img = _to_pil(vis_eo[i])
        strip = PILImage.fromarray(
            np.concatenate([np.array(sar_img), np.array(fake_img), np.array(real_img)], axis=1)
        )
        images.append(wb.wandb.Image(strip, caption=f"{name}  [SAR | Generated | GT]"))
    G.train()
    return {"vis/triplets": images}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_train", type=int, default=None, help="override for smoke runs")
    ap.add_argument("--max_val", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--wandb_mode", default=None, help="online|offline|disabled")
    ap.add_argument("--no_fid", action="store_true", help="skip FID (fast local smoke tests)")
    ap.add_argument("--no_save", action="store_true", help="skip checkpoint writes (low disk)")
    ap.add_argument("--resume", default=None,
                    help="'auto' = load last.pt from out_dir, or explicit checkpoint path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None: cfg["train"]["epochs"] = args.epochs
    if args.max_train is not None: cfg["data"]["max_train"] = args.max_train
    if args.max_val is not None: cfg["data"]["max_val"] = args.max_val
    if args.device: cfg["device"] = args.device
    if args.wandb_mode: cfg["wandb"]["mode"] = args.wandb_mode

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        # CUDA-only speedups (no-ops/unavailable on MPS). Input is a fixed 256x256, so cuDNN
        # autotune picks the fastest kernels once; TF32 gives a free matmul speedup on Ampere (A10G/A100).
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  run={cfg['run_name']}  losses={cfg['loss']}")

    train_loader, val_loader = build_loaders(cfg)
    G, D = make_models(cfg, device)

    lw = cfg["loss"]
    l1       = nn.L1Loss()
    gan_loss = GANLoss(lw.get("gan_mode", "vanilla"),
                       label_smoothing=lw.get("label_smoothing", 0.0)).to(device)
    # Hybrid cGAN loss: adv (multiscale) + L1 + VGG perceptual + classification (Wang 2022a, eq. 6).
    perc     = VGGPerceptualLoss().to(device) if lw.get("lambda_perc", 0) > 0 else None
    style    = None
    ce       = nn.CrossEntropyLoss(ignore_index=-100) if lw.get("lambda_cls", 0) > 0 else None
    classes  = cfg["model"].get("classes", list(TERRAINS))
    terr2idx = {t: i for i, t in enumerate(classes)}
    use_gan  = lw.get("lambda_gan", 0) > 0
    grad_clip = cfg["train"].get("grad_clip", 0.0)

    lr_g = cfg["train"].get("lr_g", cfg["train"]["lr"])
    lr_d = cfg["train"].get("lr_d", cfg["train"]["lr"])
    optG = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # Mixed precision (CUDA only) — ~1.5-2x faster + less VRAM on the heavy ViT generator.
    # Auto-disabled off-CUDA so local CPU/MPS smoke runs stay full precision.
    amp_on = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scalerG = torch.amp.GradScaler(enabled=amp_on)
    scalerD = torch.amp.GradScaler(enabled=amp_on)

    # ----- resume from checkpoint -----
    start_epoch, best_lpips, gstep, wandb_run_id = 1, float("inf"), 0, None
    if args.resume:
        ckpt_path = (out / "last.pt") if args.resume == "auto" else Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        _unwrap(G).load_state_dict(ckpt["G"])
        _unwrap(D).load_state_dict(ckpt["D"])
        optG.load_state_dict(ckpt["optG"])
        optD.load_state_dict(ckpt["optD"])
        start_epoch  = ckpt["epoch"] + 1
        best_lpips   = ckpt.get("best_lpips", float("inf"))
        gstep        = ckpt.get("gstep", 0)
        wandb_run_id = ckpt.get("wandb_run_id")
        print(f"Resumed from {ckpt_path} (epoch {ckpt['epoch']}, best LPIPS={best_lpips:.4f})")

    # Linear LR decay in second half of training (pix2pix standard).
    # lr stays constant for first decay_start epochs, then drops linearly to 0.
    n_epochs = cfg["train"]["epochs"]
    decay_start = int(cfg["train"].get("lr_decay_start_frac", 0.5) * n_epochs)
    def _lr_lambda(last_epoch):   # last_epoch is 0-indexed; 0 = after first step()
        ep = last_epoch + 1
        if ep <= decay_start or decay_start >= n_epochs:
            return 1.0
        return max(0.0, 1.0 - (ep - decay_start) / (n_epochs - decay_start))
    schedG = torch.optim.lr_scheduler.LambdaLR(optG, _lr_lambda)
    schedD = torch.optim.lr_scheduler.LambdaLR(optD, _lr_lambda)
    # Fast-forward scheduler to match start_epoch when resuming
    for _ in range(start_epoch - 1):
        schedG.step(); schedD.step()

    wb = WandbRun(project=cfg["wandb"]["project"], name=cfg["run_name"], config=cfg,
                  mode=cfg["wandb"].get("mode", "online"),
                  group=cfg["wandb"].get("group"), tags=cfg["wandb"].get("tags"),
                  resume_id=wandb_run_id)
    logger = LossLogger(out)

    # Cache a fixed set of val samples for consistent visual comparison across epochs.
    # Pick up to 2 samples from each terrain so every terrain is represented.
    vis_sar, vis_eo, vis_names = _cache_vis_batch(val_loader, n_per_terrain=2, device=device)

    for epoch in range(start_epoch, n_epochs + 1):
        G.train(); D.train()
        t0 = time.time()
        ep = {"g_total": 0.0, "g_gan": 0.0, "g_l1": 0.0, "g_perc": 0.0,
              "g_fm": 0.0, "g_style": 0.0, "g_cls": 0.0, "d_total": 0.0, "n": 0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{n_epochs}",
                    unit="batch", leave=False, dynamic_ncols=True)
        dt = device.type
        for batch in pbar:
            sar = batch["sar"].to(device, non_blocking=True)
            eo  = batch["eo"].to(device, non_blocking=True)
            z = torch.zeros((), device=device)
            with torch.autocast(device_type=dt, enabled=amp_on):
                fake, cls_logits = gen_forward(G, sar)

            # ----- D step (only if adversarial term active) -----
            d_val = 0.0
            if use_gan:
                optD.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dt, enabled=amp_on):
                    loss_d = 0.5 * (gan_loss(D(sar, eo), True)
                                    + gan_loss(D(sar, fake.detach()), False))
                scalerD.scale(loss_d).backward(); scalerD.step(optD); scalerD.update()
                d_val = loss_d.item()

            # ----- G step -----
            optG.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dt, enabled=amp_on):
                g_gan   = gan_loss(D(sar, fake), True) if use_gan else z
                g_l1    = l1(fake, eo)
                g_perc  = perc(fake, eo)  if perc  is not None else z
                g_style = style(fake, eo) if style is not None else z
                g_fm    = z
                if ce is not None and cls_logits is not None:    # terrain classification (Hybrid cGAN)
                    # Unknown/unlabelled terrains map to -100 (ignored). Skip if a batch is all-ignored.
                    tgt = torch.tensor([terr2idx.get(t, -100) for t in batch["terrain"]], device=device)
                    g_cls = ce(cls_logits, tgt) if (tgt != -100).any() else z
                else:
                    g_cls = z
                g_total = (lw.get("lambda_gan",   0) * g_gan
                         + lw["lambda_l1"]           * g_l1
                         + lw.get("lambda_perc",  0) * g_perc
                         + lw.get("lambda_fm",    0) * g_fm
                         + lw.get("lambda_style", 0) * g_style
                         + lw.get("lambda_cls",   0) * g_cls)
            scalerG.scale(g_total).backward()
            if grad_clip > 0:
                scalerG.unscale_(optG)
                torch.nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            scalerG.step(optG); scalerG.update()

            bs = sar.size(0); ep["n"] += bs
            ep["g_total"] += g_total.item() * bs
            ep["g_gan"]   += g_gan.detach().item()   * bs
            ep["g_l1"]    += g_l1.item()             * bs
            ep["g_perc"]  += g_perc.detach().item()  * bs
            ep["g_fm"]    += g_fm.detach().item()    * bs
            ep["g_style"] += g_style.detach().item() * bs
            ep["g_cls"]   += g_cls.detach().item()   * bs
            ep["d_total"] += d_val * bs
            gstep += 1

            pbar.set_postfix(G=f"{g_total.item():.3f}", D=f"{d_val:.3f}",
                             l1=f"{g_l1.item():.3f}", cls=f"{g_cls.detach().item():.3f}",
                             refresh=False)

            if gstep % cfg["train"].get("log_every", 50) == 0:
                wb.log({"step/g_total":  g_total.item(),
                        "step/g_gan":    g_gan.detach().item(),
                        "step/g_l1":     g_l1.item(),
                        "step/g_perc":   g_perc.detach().item(),
                        "step/g_cls":    g_cls.detach().item(),
                        "step/d_total":  d_val,
                        "global_step":   gstep}, step=gstep)
        pbar.close()

        n = max(ep["n"], 1)
        row = {"epoch": epoch, **{k: ep[k] / n for k in
               ("g_total", "g_gan", "g_l1", "g_perc", "g_fm", "g_style", "g_cls", "d_total")},
               "sec": round(time.time() - t0, 1)}

        # ----- validation (metrics + val losses) — always run on final epoch -----
        val_str = ""
        if epoch % cfg["train"].get("val_every", 1) == 0 or epoch == n_epochs:
            val = validate(G, val_loader, device, lw, perc, l1,
                           cfg["train"].get("val_metric_batches"),
                           compute_fid=cfg["train"].get("compute_fid", True) and not args.no_fid)
            row.update(val)
            fid_str = f"  FID={val['val_FID']:.2f}" if "val_FID" in val else ""
            val_str = f"  LPIPS={val['val_LPIPS']:.4f}  SSIM={val['val_SSIM']:.4f}  PSNR={val['val_PSNR']:.2f}{fid_str}"
            if val["val_LPIPS"] < best_lpips:
                best_lpips = val["val_LPIPS"]
                val_str += " *"
                if not args.no_save:
                    _atomic_save({"G": _unwrap(G).state_dict(), "D": _unwrap(D).state_dict(),
                                  "optG": optG.state_dict(), "optD": optD.state_dict(),
                                  "epoch": epoch, "best_lpips": best_lpips, "gstep": gstep,
                                  "cfg": cfg, "val": val,
                                  "wandb_run_id": wb.run.id if wb.run else None},
                                 out / "best.pt")
        if not args.no_save:
            _atomic_save({"G": _unwrap(G).state_dict(), "D": _unwrap(D).state_dict(),
                          "optG": optG.state_dict(), "optD": optD.state_dict(),
                          "epoch": epoch, "best_lpips": best_lpips, "gstep": gstep,
                          "cfg": cfg,
                          "wandb_run_id": wb.run.id if wb.run else None},
                         out / "last.pt")

        logger.log(row); logger.plot()

        wb_payload = {
            "epoch": epoch,
            # Train losses (one panel)
            "train/G_total":   row["g_total"],
            "train/D_total":   row["d_total"],
            "train/G_L1":      row["g_l1"],
            "train/G_GAN":     row["g_gan"],
            "train/G_perc":    row["g_perc"],
            "train/G_cls":     row["g_cls"],
        }
        if "val_LPIPS" in row:
            val_wb = {
                # Perceptual metrics (own panel — primary ranked metrics first)
                "val/LPIPS":   row["val_LPIPS"],
                "val/SSIM":    row["val_SSIM"],
                "val/PSNR":    row["val_PSNR"],
                # Val losses
                "val/G_total": row["val_g_total"],
                "val/G_L1":    row["val_g_l1"],
                "val/G_perc":  row["val_g_perc"],
                # Visual triplets: SAR | Generated EO | Ground Truth EO
                **_make_image_payload(G, vis_sar, vis_eo, vis_names, wb),
                # Best tracker
                "best/LPIPS":  best_lpips,
            }
            if "val_FID" in row:
                val_wb["val/FID"] = row["val_FID"]
            wb_payload.update(val_wb)
        wb.log(wb_payload, step=gstep)

        schedG.step(); schedD.step()
        cur_lr_g = schedG.get_last_lr()[0]

        print(f"[{epoch:03d}/{n_epochs}]  G={row['g_total']:.4f}  D={row['d_total']:.4f}"
              f"  l1={row['g_l1']:.4f}  perc={row['g_perc']:.4f}  cls={row['g_cls']:.4f}"
              f"  lr={cur_lr_g:.2e}  t={row['sec']:.0f}s{val_str}", flush=True)

    wb.summary({"best_val_LPIPS": best_lpips})
    wb.finish()
    ckpt = "(not saved: --no_save)" if args.no_save else f"-> {out/'best.pt'}"
    print(f"done. best val LPIPS={best_lpips:.4f}  {ckpt}", flush=True)


if __name__ == "__main__":
    main()
