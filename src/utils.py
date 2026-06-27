"""Shared utilities: seeding, device selection, config, loss logging, and a thin W&B wrapper."""
from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device(prefer="auto") -> torch.device:
    if prefer not in ("auto", None):
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


class LossLogger:
    """Persists per-epoch losses/metrics to CSV+JSON and renders the loss curve (assignment §5.1)."""

    def __init__(self, out_dir: str | Path):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def log(self, row: dict):
        self.rows.append(row)
        keys = sorted({k for r in self.rows for k in r})
        with open(self.out / "losses.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(self.rows)
        (self.out / "losses.json").write_text(json.dumps(self.rows, indent=2))

    def plot(self, fname="loss_curve.png"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [r["epoch"] for r in self.rows]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for key, lab in [("g_total", "G total (train)"), ("d_total", "D (train)"),
                         ("val_g_total", "G total (val)")]:
            ys = [r.get(key) for r in self.rows]
            if any(y is not None for y in ys):
                ax[0].plot(ep, ys, marker=".", label=lab)
        ax[0].set_title("Adversarial training losses"); ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
        for key, lab in [("val_LPIPS", "val LPIPS ↓"), ("val_SSIM", "val SSIM ↑")]:
            ys = [r.get(key) for r in self.rows]
            if any(y is not None for y in ys):
                ax[1].plot(ep, ys, marker=".", label=lab)
        ax[1].set_title("Validation metrics"); ax[1].set_xlabel("epoch")
        if ax[1].get_lines(): ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(self.out / fname, dpi=120); plt.close(fig)


class WandbRun:
    """Optional W&B logging. Falls back to disabled if wandb/key absent so code always runs.

    mode: 'online' | 'offline' | 'disabled'. Set WANDB_MODE env or config to control.
    """

    def __init__(self, project: str, name: str, config: dict, mode: str = "online",
                 group: str | None = None, tags: list | None = None,
                 resume_id: str | None = None):
        self.run = None
        try:
            import wandb
            if not (os.environ.get("WANDB_API_KEY") or mode in ("offline", "disabled")):
                mode = "offline"  # no key -> log locally rather than crash
            self.wandb = wandb
            init_kwargs = dict(project=project, name=name, config=config, mode=mode,
                               group=group, tags=tags, reinit="finish_previous")
            if resume_id:
                init_kwargs.update(resume="allow", id=resume_id)
            self.run = wandb.init(**init_kwargs)
            self._define_metrics()
        except Exception as e:  # pragma: no cover
            print(f"[wandb] disabled ({e}); continuing without it.")

    def _define_metrics(self):
        if self.run is None:
            return
        w = self.wandb
        # epoch as x-axis for all train/val panels
        w.define_metric("epoch")
        for prefix in ("train", "val", "best"):
            w.define_metric(f"{prefix}/*", step_metric="epoch")
        # step-level losses use global step
        w.define_metric("step/*", step_metric="global_step")

    def log(self, data: dict, step: int | None = None):
        if self.run is not None:
            self.wandb.log(data, step=step)

    def summary(self, data: dict):
        if self.run is not None:
            for k, v in data.items():
                self.run.summary[k] = v

    def finish(self):
        if self.run is not None:
            self.wandb.finish()
