"""Build the final combined dataset (Kaggle terrain set + SEN12MS subset) and carve a scene-disjoint
train/val/test split. Reuses the sampling in scripts/build_dataset.py.

    python scripts/build_final_dataset.py --v2_root v_2 --download --seasons spring \
        --sen12ms_root data/sen12ms_raw --out data/final_dataset --target_total 18000 --k_per_scene 40

Output: data/final_dataset/{train,val,test}/<terrain>/{s1,s2}/<name>.png. Needs rasterio for SEN12MS
GeoTIFFs. SEN12MS downloads from the TU Munich mediaTUM dataserv; pass --url_base if the default 404s.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

# Allow `python scripts/build_final_dataset.py` from the repo root to import src/ and the builder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_dataset import group_scenes, make_iters, sample_plan, write_pool  # noqa: E402
from src.data.dataset import make_split_dirs                                          # noqa: E402

# SEN12MS season -> ROI tag (standard dataset naming). Each season ships s1/s2/lc tarballs.
SEASON_TAG = {"spring": "ROIs1158", "summer": "ROIs1868", "fall": "ROIs1970", "winter": "ROIs2017"}
# Single-file download pattern for the mediaTUM dataserv share (id m1474000). VERIFY if it changes.
DEFAULT_URL_BASE = "https://dataserv.ub.tum.de/s/m1474000/download?path=%2F&files="


def download_sen12ms(seasons: list[str], dest: Path, url_base: str):
    """Download + extract the s1/s2/lc tarballs for the requested SEN12MS seasons into `dest`.
    Skips any tarball already extracted. Uses streaming requests with a progress bar."""
    import requests
    from tqdm import tqdm

    dest.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        tag = SEASON_TAG.get(season)
        if tag is None:
            print(f"  ! unknown season '{season}' (expected one of {list(SEASON_TAG)}); skipping")
            continue
        for modality in ("lc", "s1", "s2"):                # lc/s1 are small; s2 (13-band) is the big one
            stem = f"{tag}_{season}_{modality}"
            if (dest / f"{tag}_{season}").is_dir():         # already extracted this season -> skip all
                print(f"  = {tag}_{season} already extracted; skipping {modality}")
                continue
            tar_path = dest / f"{stem}.tar.gz"
            url = f"{url_base}{stem}.tar.gz"
            if not tar_path.exists():
                print(f"  -> downloading {stem}.tar.gz\n     {url}")
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0))
                    with open(tar_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk); bar.update(len(chunk))
            print(f"  -> extracting {tar_path.name}")
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(dest)
            tar_path.unlink()                               # remove tarball after extract to save disk


def cleanup(paths: list[Path]):
    """Delete intermediate / now-redundant data dirs after the final set is verified."""
    for p in paths:
        p = Path(p)
        if p.exists():
            print(f"  rm -rf {p}")
            shutil.rmtree(p) if p.is_dir() else p.unlink()


def _verify_final(out: Path) -> bool:
    """Confirm out/{train,val,test} each hold paired PNGs before we delete anything."""
    from src.data.dataset import index_dataset
    ok = True
    for sp in ("train", "val", "test"):
        n = len(index_dataset(out / sp)) if (out / sp).exists() else 0
        print(f"  {sp:5s}: {n} pairs")
        ok = ok and n > 0
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_root", default="v_2", help="Kaggle v_2 set (4 terrains)")
    ap.add_argument("--sen12ms_root", default="data/sen12ms_raw", help="SEN12MS root (downloaded/extracted here)")
    ap.add_argument("--out", default="data/final_dataset", help="final dataset (holds train/val/test)")
    ap.add_argument("--download", action="store_true", help="download the SEN12MS season subset first")
    ap.add_argument("--seasons", nargs="+", default=["spring"], help="SEN12MS seasons to use (breadth vs disk)")
    ap.add_argument("--url_base", default=DEFAULT_URL_BASE, help="SEN12MS single-file download base URL")
    ap.add_argument("--k_per_scene", type=int, default=40, help="max patches/scene (scene-split makes this leak-safe)")
    ap.add_argument("--target_total", type=int, default=18000, help="approx total pairs (train+val+test)")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--sar_clip", type=float, nargs=2, default=(-25.0, 0.0), help="SEN12MS VV dB clip")
    ap.add_argument("--s2_clip", type=float, default=3000.0, help="SEN12MS S2 reflectance clip")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_quality_filter", action="store_true")
    ap.add_argument("--cleanup", action="store_true", help="after build+verify, delete pool/raw SEN12MS/v_2")
    ap.add_argument("--cleanup_only", action="store_true", help="skip build; just delete intermediates")
    args = ap.parse_args()

    out = Path(args.out)
    pool = out.parent / "_final_pool"          # un-split staging dir (deleted after the split)
    redundant = [pool, Path(args.sen12ms_root), Path(args.v2_root),
                 Path("v_2_split"), Path("v_2_split_scene")]

    if args.cleanup_only:
        if not _verify_final(out):
            raise SystemExit("Refusing to clean up: final_dataset is incomplete/empty.")
        cleanup(redundant)
        print("cleanup done."); return

    # 1) optional SEN12MS subset download
    if args.download:
        print(f"Downloading SEN12MS seasons={args.seasons} -> {args.sen12ms_root}")
        download_sen12ms(args.seasons, Path(args.sen12ms_root), args.url_base)

    # 2) balanced, breadth-first sampling across v_2 + SEN12MS (VV only; terrain-balanced)
    rng = np.random.default_rng(args.seed)
    sen12ms_root = args.sen12ms_root if Path(args.sen12ms_root).exists() else None
    if sen12ms_root is None:
        print(f"WARNING: {args.sen12ms_root} not found — building from v_2 only (no SEN12MS breadth).")
    iters = make_iters(v2_root=args.v2_root, sen12ms_root=sen12ms_root)
    plan = sample_plan(group_scenes(iters), args.k_per_scene, args.target_total, rng)

    # 3) write the un-split pool, then carve a SCENE-DISJOINT train/val/test into `out`
    if pool.exists():
        shutil.rmtree(pool)
    write_pool(plan, pool, args.sar_clip, args.s2_clip,
               quality_filter=not args.no_quality_filter, seed=args.seed, k_per_scene=args.k_per_scene)
    print(f"Splitting pool -> {out} (scene-disjoint; copies, not symlinks, so it tars cleanly)")
    make_split_dirs(pool, out, strategy="scene", val_frac=args.val_frac, test_frac=args.test_frac,
                    seed=args.seed, use_symlinks=False)

    print("\nFinal dataset:")
    ok = _verify_final(out)
    if args.cleanup:
        if not ok:
            raise SystemExit("Refusing to clean up: final_dataset is incomplete/empty.")
        cleanup(redundant)
    else:
        print(f"\nKept intermediates. After verifying, reclaim disk with:\n"
              f"  python scripts/build_final_dataset.py --cleanup_only --out {out} "
              f"--sen12ms_root {args.sen12ms_root} --v2_root {args.v2_root}")
    print(f"\nNext: tar czf /tmp/final_dataset.tar.gz -C {out.parent} {out.name}  &&  "
          f"modal run modal_app.py::upload --local-dir {out}")


if __name__ == "__main__":
    main()
