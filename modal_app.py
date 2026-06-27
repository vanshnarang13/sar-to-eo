"""Modal cloud-GPU entrypoints for training + evaluation, with live logs and W&B online tracking."""
from __future__ import annotations

import modal
import yaml

app = modal.App("galaxeye-sar2eo")

_base = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        # Cache model weights at image build time so containers don't download each run
        "python -c \"import lpips; lpips.LPIPS(net='alex', verbose=False)\"",          # LPIPS AlexNet (~233 MB)
        "python -c \"import torchvision; torchvision.models.vgg16(weights='IMAGENET1K_V1')\"",  # VGG16 perceptual (~528 MB)
        "python -c \"from torchmetrics.image import FrechetInceptionDistance; FrechetInceptionDistance(feature=2048, normalize=True)\"",  # InceptionV3 for FID (~91 MB)
    )
    .env({"PYTHONUNBUFFERED": "1"})   # unbuffered stdout -> prints stream live to Modal logs (cheap last layer)
)
# Train/eval image = model code only. Build image additionally needs rasterio (SEN12MS GeoTIFFs).
image = _base.add_local_python_source("src", "infer", "scripts")
build_image = _base.pip_install("rasterio").add_local_python_source("src", "infer", "scripts")

vol = modal.Volume.from_name("sar2eo-vol", create_if_missing=True)
VOL = "/vol"
# W&B key injected from the Modal secret you create with `modal secret create wandb ...`
wandb_secret = modal.Secret.from_name("wandb")


def _split_root(cfg: dict) -> str:
    """Volume path of the split dir this config trains/evals on (set by data.volume_split)."""
    return f"{VOL}/{cfg['data'].get('volume_split', 'v_2_split')}"


def _patch_cfg(cfg: dict) -> str:
    """Point a config at the Volume paths and write it to a temp file inside the container."""
    from pathlib import Path
    d = cfg["data"]
    if "train_root" in d:
        root = _split_root(cfg)
        d["train_root"] = f"{root}/train"
        d["val_root"]   = f"{root}/val"
        d["test_root"]  = f"{root}/test"
    else:
        d["root"] = f"{VOL}/v_2"
    cfg["out_dir"] = f"{VOL}/outputs/runs/{cfg['run_name']}"
    path = f"/tmp/{cfg['run_name']}.yaml"
    Path(path).write_text(yaml.safe_dump(cfg))
    return path


@app.function(image=image, gpu="A10G", volumes={VOL: vol}, secrets=[wandb_secret],
              timeout=24 * 3600)
def _train_remote(cfg: dict, extra_args: list[str]):
    import sys
    path = _patch_cfg(cfg)
    sys.argv = ["train", "--config", path] + list(extra_args)
    from src.train import main
    main()
    vol.commit()


@app.function(image=image, gpu="A100-40GB", volumes={VOL: vol}, secrets=[wandb_secret],
              timeout=24 * 3600)
def _train_remote_a100(cfg: dict, extra_args: list[str]):
    import sys
    path = _patch_cfg(cfg)
    sys.argv = ["train", "--config", path] + list(extra_args)
    from src.train import main
    main()
    vol.commit()


@app.function(image=image, gpu="T4", volumes={VOL: vol}, secrets=[wandb_secret],
              timeout=24 * 3600)
def _train_remote_t4(cfg: dict, extra_args: list[str]):
    import sys
    path = _patch_cfg(cfg)
    sys.argv = ["train", "--config", path] + list(extra_args)
    from src.train import main
    main()
    vol.commit()


@app.function(image=image, gpu="A10G", volumes={VOL: vol}, timeout=6 * 3600)
def _eval_remote(cfg: dict, which: str, no_fid: bool):
    import json
    from pathlib import Path
    from src.eval import evaluate_checkpoint_on_dir
    weights = f"{VOL}/outputs/runs/{cfg['run_name']}/best.pt"
    # Evaluate on the physical split dir so train/val/test share one leak-free partition.
    res = evaluate_checkpoint_on_dir(weights, f"{_split_root(cfg)}/{which}", compute_fid=not no_fid)
    print(json.dumps(res, indent=2))
    out = Path(f"{VOL}/outputs/runs/{cfg['run_name']}/eval_{which}.json")
    out.write_text(json.dumps(res, indent=2)); vol.commit()


@app.function(image=image, gpu="H100", volumes={VOL: vol}, secrets=[wandb_secret],
              timeout=24 * 3600)
def _train_remote_h100(cfg: dict, extra_args: list[str]):
    import sys
    path = _patch_cfg(cfg)
    sys.argv = ["train", "--config", path] + list(extra_args)
    from src.train import main
    main()
    vol.commit()


_TRAIN_FNS = {"A10G": _train_remote, "H100": _train_remote_h100,
              "A100-40GB": _train_remote_a100, "T4": _train_remote_t4}


@app.local_entrypoint()
def train(config: str, gpu: str = "A10G", epochs: int = 0, max_train: int = 0,
          max_val: int = 0, wandb_mode: str = "online", resume: bool = False,
          resume_from: str = "", no_fid: bool = False, no_save: bool = False):
    """Read the config locally, launch GPU training on Modal (live logs).

    Resume example (latest checkpoint, out_dir/last.pt):
        modal run modal_app.py::train --config configs/pix2pix_baseline.yaml --resume
    Resume from a specific checkpoint (e.g. best.pt if last.pt got corrupted on interrupt):
        modal run modal_app.py::train --config configs/hybrid_cgan_swin.yaml --gpu H100 \\
            --epochs 200 --resume-from /vol/outputs/runs/hybrid_cgan_swin/best.pt
    Smoke example (cheap, skips FID + caps val so it stays fast on an expensive GPU):
        modal run modal_app.py::train --config configs/hybrid_cgan_swin.yaml \\
            --gpu H100 --epochs 1 --max-train 64 --max-val 32 --no-fid --no-save \\
            --wandb-mode disabled
    """
    cfg = yaml.safe_load(open(config))
    cfg.setdefault("wandb", {})["mode"] = wandb_mode
    extra = []
    if epochs:
        extra += ["--epochs", str(epochs)]
    if max_train:
        extra += ["--max_train", str(max_train)]
    if max_val:
        extra += ["--max_val", str(max_val)]
    if resume_from:
        extra += ["--resume", resume_from]
    elif resume:
        extra += ["--resume", "auto"]
    if no_fid:
        extra += ["--no_fid"]
    if no_save:
        extra += ["--no_save"]
    _TRAIN_FNS.get(gpu, _train_remote).remote(cfg, extra)


@app.local_entrypoint()
def evaluate(config: str, which: str = "test", gpu: str = "A10G", no_fid: bool = False):
    """Evaluate best.pt on a physical split dir (which = train|val|test)."""
    cfg = yaml.safe_load(open(config))
    _eval_remote.remote(cfg, which, no_fid)


@app.function(image=image, volumes={VOL: vol}, timeout=3600)
def _extract_tar(tar_path: str, dest: str):
    import tarfile, os
    size_mb = os.path.getsize(tar_path) / 1e6
    print(f"Extracting {tar_path} ({size_mb:.0f} MB) -> {dest} ...")   # size = integrity check
    with tarfile.open(tar_path, "r:*") as tf:        # r:* auto-detects gz / plain tar
        tf.extractall(dest)
    vol.commit()
    print("Extraction done.")


@app.function(image=image, volumes={VOL: vol}, timeout=3600)
def _pack_tar(src_dir: str, tar_path: str):
    import tarfile, os
    mode = "w:gz" if tar_path.endswith(".gz") else "w"
    print(f"Packing {src_dir} -> {tar_path} (mode={mode}) ...")
    with tarfile.open(tar_path, mode) as tf:
        tf.add(src_dir, arcname=os.path.basename(src_dir))
    size_mb = os.path.getsize(tar_path) / 1e6
    vol.commit()
    print(f"Packed {size_mb:.0f} MB -> {tar_path}")


@app.local_entrypoint()
def pack_volume(src: str = "final_dataset", compress: bool = False):
    """Tar a directory on the Modal volume so you can `modal volume get` it.

        modal run modal_app.py::pack_volume --src final_dataset              # fast, no compression
        modal run modal_app.py::pack_volume --src final_dataset --compress   # smaller but slow
        modal volume get sar2eo-vol /final_dataset.tar ./final_dataset.tar
    """
    ext = ".tar.gz" if compress else ".tar"
    _pack_tar.remote(f"{VOL}/{src}", f"{VOL}/{src}{ext}")


@app.local_entrypoint()
def unpack_volume(tar_name: str = "final_dataset.tar.gz"):
    """Extract a tarball already on the Modal volume.

        modal volume put sar2eo-vol ./final_dataset.tar.gz /final_dataset.tar.gz
        modal run modal_app.py::unpack_volume --tar-name final_dataset.tar.gz
    """
    _extract_tar.remote(f"{VOL}/{tar_name}", VOL)


@app.function(image=image, volumes={VOL: vol}, timeout=1800)
def _count_dataset(tar_name: str, split_dir: str):
    """Count SAR/EO pairs per split & terrain. From a tarball (reads its index -> also verifies the
    tar isn't truncated) when tar_name is given, else from the already-extracted dir on the volume."""
    import os, tarfile
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(lambda: {"s1": 0, "s2": 0}))

    if tar_name:
        path = f"{VOL}/{tar_name}"
        print(f"Counting from tarball {path} (also verifies it is not truncated) ...", flush=True)
        with tarfile.open(path, "r:*") as tf:
            for m in tf:                                  # iterates member headers (no extract to disk)
                if not (m.isfile() and m.name.endswith(".png")):
                    continue
                parts = m.name.split("/")                  # final_dataset/<split>/<terrain>/<s1|s2>/f.png
                if len(parts) >= 5 and parts[3] in ("s1", "s2"):
                    counts[parts[1]][parts[2]][parts[3]] += 1
    else:
        src = f"{VOL}/{split_dir}"
        print(f"Counting from extracted dir {src} ...", flush=True)
        for split in ("train", "val", "test"):
            base = os.path.join(src, split)
            if not os.path.isdir(base):
                continue
            for terrain in sorted(os.listdir(base)):
                for mod in ("s1", "s2"):
                    d = os.path.join(base, terrain, mod)
                    if os.path.isdir(d):
                        counts[split][terrain][mod] = sum(f.endswith(".png") for f in os.listdir(d))

    grand = {}
    for split in ("train", "val", "test"):
        if split not in counts:
            continue
        print(f"\n=== {split} ===", flush=True)
        stot = 0
        for terrain in sorted(counts[split]):
            c = counts[split][terrain]
            stot += c["s1"]
            flag = "" if c["s1"] == c["s2"] else f"  !! s1 != s2 ({c['s1']} vs {c['s2']})"
            print(f"  {terrain:12s} pairs={c['s1']:6d}{flag}", flush=True)
        grand[split] = stot
        print(f"  {'TOTAL':12s} pairs={stot:6d}", flush=True)

    total = sum(grand.values()) or 1
    print(f"\nSplit totals: " + "  ".join(f"{k}={grand.get(k,0)}" for k in ("train", "val", "test"))
          + f"  (all={sum(grand.values())})", flush=True)
    print("Ratios:       " + "  ".join(f"{k}={grand.get(k,0)/total:.3f}" for k in ("train", "val", "test")),
          flush=True)


@app.local_entrypoint()
def count_dataset(tar_name: str = "", split_dir: str = "final_dataset"):
    """Report #pairs per split & terrain on the Modal volume.

        modal run modal_app.py::count_dataset                                # from extracted dir
        modal run modal_app.py::count_dataset --tar-name final_dataset.tar   # from tarball (+verify)
    """
    _count_dataset.remote(tar_name, split_dir)


@app.local_entrypoint()
def upload(local_dir: str = "v_2"):
    """Pack dataset into a single tar.gz and upload it (much faster than 32k individual files)."""
    import os
    import tarfile
    tar_path = f"/tmp/{local_dir}.tar.gz"
    print(f"Packing {local_dir} -> {tar_path} ...")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(local_dir, arcname=local_dir)
    size_mb = os.path.getsize(tar_path) / 1e6
    print(f"Packed {size_mb:.0f} MB. Uploading single file ...")
    with vol.batch_upload(force=True) as b:
        b.put_file(tar_path, f"/{local_dir}.tar.gz")
    print("Upload done. Extracting on Modal ...")
    _extract_tar.remote(f"{VOL}/{local_dir}.tar.gz", VOL)
    print(f"Dataset ready at Volume:{VOL}/{local_dir}")


# ---------------------------------------------------------------------------
#  SEN12MS download + final dataset build (all on Modal, no local disk needed)
# ---------------------------------------------------------------------------
# SEN12MS season -> ROI tag (standard naming from the dataset paper).
_SEN12MS_SEASONS = {
    "spring": "ROIs1158_spring", "summer": "ROIs1868_summer",
    "fall":   "ROIs1970_fall",   "winter": "ROIs2017_winter",
}
_SEN12MS_URL_BASE = "https://dataserv.ub.tum.de/s/m1474000/download?path=%2F&files="


@app.function(image=build_image, volumes={VOL: vol}, timeout=12 * 3600)
def _download_sen12ms_remote(seasons: list[str], url_base: str):
    """Download + extract SEN12MS season tarballs (s1/s2/lc) to {VOL}/sen12ms_raw on the volume."""
    import tarfile
    from pathlib import Path
    import requests
    dest = Path(f"{VOL}/sen12ms_raw")
    dest.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        tag = _SEN12MS_SEASONS.get(season)
        if tag is None:
            print(f"  ! unknown season '{season}'; skipping"); continue
        for mod in ("lc", "s1", "s2"):
            marker = dest / f".done_{tag}_{mod}"
            if marker.exists():
                print(f"  = {tag}_{mod} already done; skipping"); continue
            fname = f"{tag}_{mod}.tar.gz"
            tar_path = dest / fname
            url = f"{url_base}{fname}"
            print(f"  -> downloading {fname}\n     {url}")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                with open(tar_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk); done += len(chunk)
                        if total and done % (100 << 20) < (1 << 20):
                            print(f"     {done/1e9:.1f}/{total/1e9:.1f} GB")
            print(f"  -> extracting {fname}")
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(dest)
            tar_path.unlink()
            marker.touch()
            vol.commit()
    print(f"SEN12MS seasons {seasons} ready at {dest}")
    vol.commit()


@app.function(image=build_image, volumes={VOL: vol}, timeout=6 * 3600)
def _build_final_remote(k_per_scene: int, target_total: int, val_frac: float, test_frac: float,
                        seed: int):
    """Combine v_2 + SEN12MS -> balanced, scene-disjoint final_dataset on the volume."""
    import shutil
    from pathlib import Path
    import numpy as np
    from scripts.build_dataset import group_scenes, make_iters, sample_plan, write_pool
    from src.data.dataset import make_split_dirs, index_dataset

    sen12ms = f"{VOL}/sen12ms_raw" if Path(f"{VOL}/sen12ms_raw").exists() else None
    v2 = f"{VOL}/v_2" if Path(f"{VOL}/v_2").exists() else None
    if not v2 and not sen12ms:
        raise SystemExit("Neither v_2 nor sen12ms_raw found on the volume.")

    rng = np.random.default_rng(seed)
    iters = make_iters(v2_root=v2, sen12ms_root=sen12ms)
    scenes = group_scenes(iters)
    plan = sample_plan(scenes, k_per_scene, target_total, rng)

    pool = Path(f"{VOL}/_final_pool")
    if pool.exists(): shutil.rmtree(pool)
    import os
    n_cpu = min(os.cpu_count() or 4, 16)
    write_pool(plan, pool, (-25.0, 0.0), 3000.0,
               quality_filter=True, seed=seed, k_per_scene=k_per_scene, n_workers=n_cpu)

    out = Path(f"{VOL}/final_dataset")
    if out.exists(): shutil.rmtree(out)
    make_split_dirs(pool, out, strategy="scene", val_frac=val_frac, test_frac=test_frac,
                    seed=seed, use_symlinks=False)

    shutil.rmtree(pool)
    print(f"\nFinal dataset at {out}:")
    for sp in ("train", "val", "test"):
        n = len(index_dataset(out / sp)) if (out / sp).exists() else 0
        print(f"  {sp:5s}: {n} pairs")
    vol.commit()


@app.local_entrypoint()
def download_sen12ms(seasons: str = "spring", url_base: str = _SEN12MS_URL_BASE):
    """Download SEN12MS season tarballs directly to the Modal volume (no local disk needed).

    Each season is ~24 GB (s1 ~3 GB + s2 ~20 GB + lc ~0.5 GB). Skips already-downloaded modalities.

        modal run modal_app.py::download_sen12ms                        # spring only (~24 GB)
        modal run modal_app.py::download_sen12ms --seasons spring,summer  # two seasons (~48 GB)
    """
    _download_sen12ms_remote.remote([s.strip() for s in seasons.split(",")], url_base)


@app.local_entrypoint()
def build_final(k_per_scene: int = 800, target_total: int = 22000,
                val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42):
    """Build the FINAL combined dataset (v_2 + SEN12MS) with scene-disjoint split on Modal.

    Prereqs: v_2 uploaded (modal run modal_app.py::upload), SEN12MS downloaded
    (modal run modal_app.py::download_sen12ms). Output: {VOL}/final_dataset/{train,val,test}.

    Full pipeline:
        modal run modal_app.py::upload --local-dir data/v_2          # already done
        modal run modal_app.py::download_sen12ms --seasons spring    # ~24 GB, ~30 min
        modal run modal_app.py::build_final                          # combine + split
        modal run modal_app.py::verify_split --src final_dataset --strategy scene
        modal run modal_app.py::train --config configs/hybrid_cgan.yaml
    """
    _build_final_remote.remote(k_per_scene, target_total, val_frac, test_frac, seed)


@app.function(image=image, volumes={VOL: vol}, timeout=1800)
def _make_split_remote(strategy: str = "random", val_frac: float = 0.1,
                       test_frac: float = 0.1, seed: int = 42, src: str = "v_2"):
    from src.data.dataset import make_split_dirs
    out = f"{VOL}/{src}_split" if strategy == "random" else f"{VOL}/{src}_split_{strategy}"
    make_split_dirs(f"{VOL}/{src}", out, strategy=strategy,
                    val_frac=val_frac, test_frac=test_frac, seed=seed, use_symlinks=True)
    vol.commit()


@app.function(image=image, volumes={VOL: vol}, timeout=900)
def _verify_split_remote(strategy: str = "random", src: str = "v_2"):
    import json
    from pathlib import Path
    from src.data.dataset import index_dataset
    # If src itself already contains train/val/test (baked-in split from build_final), use it directly.
    direct = Path(f"{VOL}/{src}")
    if (direct / "train").is_dir():
        root = direct
    else:
        root = Path(f"{VOL}/{src}_split" if strategy == "random" else f"{VOL}/{src}_split_{strategy}")
    if not root.exists():
        print(f"MISSING: {root} — run make_split first."); return
    man = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    print(f"== {root}  strategy={man.get('strategy', '?')} ==")
    scenes = {}
    for sp in ("train", "val", "test"):
        d = root / sp
        if not d.exists():
            continue
        samples = index_dataset(d)
        terrs = sorted({s.terrain for s in samples})
        per_t = {t: sum(1 for s in samples if s.terrain == t) for t in terrs}
        scenes[sp] = {s.scene for s in samples}
        print(f"  {sp:5s}: {len(samples):6d} pairs  scenes={len(scenes[sp]):3d}  per-terrain={per_t}")
    if {"train", "val", "test"} <= set(scenes):
        tr, va, te = scenes["train"], scenes["val"], scenes["test"]
        print(f"  scene overlap  train∩val={len(tr & va)}  train∩test={len(tr & te)}  val∩test={len(va & te)}")
        print(f"  SCENE-DISJOINT: {not (tr & va or tr & te or va & te)}")


@app.local_entrypoint()
def verify_split(strategy: str = "random", src: str = "v_2"):
    """Print split counts, per-terrain balance, and scene-disjointness for the volume's split."""
    _verify_split_remote.remote(strategy=strategy, src=src)


@app.local_entrypoint()
def make_split(strategy: str = "random", val_frac: float = 0.1, test_frac: float = 0.1,
               seed: int = 42, src: str = "v_2"):
    """Build terrain-stratified train/val/test dirs on the Modal volume (run once per strategy).

    strategy='random' -> v_2_split/{train,val,test}        (8:1:1, comparable to survey Table 2)
    strategy='scene'  -> v_2_split_scene/{train,val,test}  (scene-disjoint, honest generalisation)

    Order:
        modal run modal_app.py::upload
        modal run modal_app.py::make_split                      # random split
        modal run modal_app.py::make_split --strategy scene     # scene-disjoint split
        modal run modal_app.py::train ...                       # point cfg data roots at the split
        modal run modal_app.py::evaluate --config <cfg> --which test
    """
    _make_split_remote.remote(strategy=strategy, val_frac=val_frac, test_frac=test_frac,
                              seed=seed, src=src)
