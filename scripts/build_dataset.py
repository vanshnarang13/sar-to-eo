"""Build the unified SAR(VV)->EO(RGB) dataset from the Kaggle terrain set plus SEN12MS.

Sampling is breadth-first (capped patches per scene, round-robin across scenes) and terrain-balanced,
with cross-source dedup by (roi, season, scene, patch). SAR = VV only, dB-scaled, min-max to uint8
[0,255]; optical = SEN12MS bands B4/B3/B2 as 8-bit RGB. Output is ready for make_split_dirs(scene):

    <out>/<terrain>/{s1,s2}/<source>_<season>_<scene>_p<patch>.png   (+ build_manifest.json)

SEN12MS reading needs rasterio (GeoTIFF). Run where the data lives, not on a small local disk.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------- #
#  Terrain taxonomy + MODIS IGBP(17-class) -> our 7 terrains
# ----------------------------------------------------------------------------- #
TERRAINS = ("agri", "barren", "grass", "urban", "forest", "water", "shrub")
IGBP_TO_TERRAIN = {
    1: "forest", 2: "forest", 3: "forest", 4: "forest", 5: "forest",
    6: "shrub", 7: "shrub", 8: "shrub", 9: "shrub",            # shrublands + savannas
    10: "grass",
    11: "water",                                                # permanent wetlands
    12: "agri", 14: "agri",                                     # cropland (+ mosaic)
    13: "urban",
    15: None,                                                   # snow/ice -> excluded by choice
    16: "barren",
    17: "water",
}
# The Kaggle set already uses these folder names (barrenland->barren, grassland->grass)
V2_TERRAIN_ALIAS = {"barrenland": "barren", "grassland": "grass", "agri": "agri", "urban": "urban"}
SEASONS = ("spring", "summer", "fall", "winter")
_ROI_RE = re.compile(r"(ROIs\d+)_([a-z]+)")


# ----------------------------------------------------------------------------- #
#  Preprocessing (CONTRACT-CRITICAL — these are documented design choices)
# ----------------------------------------------------------------------------- #
def sar_db_to_uint8(vv: np.ndarray, clip_lo: float = -25.0, clip_hi: float = 0.0) -> np.ndarray:
    """SEN12MS VV is float dB. Clip to [clip_lo, clip_hi] dB then min-max -> [0,255] uint8,
    matching the contract's 'dB-scaled and min-max normalised to [0,255]'."""
    vv = np.clip(vv, clip_lo, clip_hi)
    vv = (vv - clip_lo) / (clip_hi - clip_lo)
    return (vv * 255.0).round().astype(np.uint8)


def s2_bands_to_rgb(b4: np.ndarray, b3: np.ndarray, b2: np.ndarray,
                    clip: float = 3000.0) -> np.ndarray:
    """SEN12MS optical is 13-band reflectance (~0..10000). RGB = B4/B3/B2, clipped & scaled."""
    rgb = np.stack([b4, b3, b2], axis=-1).astype(np.float32)
    rgb = np.clip(rgb, 0, clip) / clip
    return (rgb * 255.0).round().astype(np.uint8)


def is_cloudy(rgb: np.ndarray, white_thresh: int = 230, frac: float = 0.5) -> bool:
    """Heuristic cloud/over-bright reject: too many near-white, low-colour-variance pixels."""
    bright = (rgb.mean(axis=-1) > white_thresh).mean()
    return bright > frac


def is_degenerate_sar(sar: np.ndarray) -> bool:
    """Reject near-constant / near-black SAR tiles (no usable structure)."""
    return float(sar.std()) < 3.0


# ----------------------------------------------------------------------------- #
#  Per-source iterators: each yields dicts with a global scene key + arrays
# ----------------------------------------------------------------------------- #
def _season_roi(name: str):
    m = _ROI_RE.search(name)
    return (m.group(1), m.group(2)) if m else ("ROIunknown", "unknown")


def iter_v2(root: Path):
    """Kaggle v_2: <root>/<terrain>/{s1,s2}/ROIs..._s{1,2}_<scene>_p<patch>.png (already contract-form)."""
    for terr_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        terrain = V2_TERRAIN_ALIAS.get(terr_dir.name, terr_dir.name)
        s1d, s2d = terr_dir / "s1", terr_dir / "s2"
        if not s1d.is_dir():
            continue
        s2map = {p.name.replace("_s2_", "_s1_"): p for p in s2d.glob("*.png")}
        for sar in sorted(s1d.glob("*.png")):
            eo = s2map.get(sar.name)
            if eo is None:
                continue
            roi, season = _season_roi(sar.name)
            sc = re.search(r"_s1_(\d+)_p(\d+)", sar.name)
            scene, patch = (sc.group(1), sc.group(2)) if sc else ("0", "0")
            yield {"source": "v2", "season": season, "roi": roi, "scene": scene, "patch": patch,
                   "terrain": terrain, "sar_png": sar, "eo_png": eo}


def iter_sen12(root: Path):
    """SEN1-2: season ROI folders with s1_<scene>/ and s2_<scene>/ PNG patches (VV + RGB, 8-bit).
    No land-cover labels -> terrain unknown (assigned later only if a label source exists)."""
    for roi_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("ROIs")):
        roi, season = _season_roi(roi_dir.name)
        for s1d in sorted(roi_dir.glob("s1_*")):
            scene = s1d.name.split("_", 1)[1]
            s2d = roi_dir / f"s2_{scene}"
            if not s2d.is_dir():
                continue
            for sar in sorted(s1d.glob("*.png")):
                eo = s2d / sar.name.replace("_s1_", "_s2_")
                if not eo.exists():
                    continue
                patch = re.search(r"_p?(\d+)\.png$", sar.name)
                yield {"source": "sen12", "season": season, "roi": roi, "scene": scene,
                       "patch": patch.group(1) if patch else "0", "terrain": None,
                       "sar_png": sar, "eo_png": eo}


def iter_sen12ms(root: Path):
    """SEN12MS: season ROI folders with s1_<scene>/, s2_<scene>/, lc_<scene>/ GeoTIFFs.
    s1 = 2-band float dB (VV,VH) -> use VV; s2 = 13-band reflectance -> B4/B3/B2;
    lc band 1 = IGBP -> majority class -> terrain. Needs rasterio.

    SPEED: terrain is determined ONCE per scene (read first lc file only) rather than per patch,
    since patches within a scene share the same geographic area. ~100x fewer rasterio opens."""
    import rasterio  # local import: only SEN12MS needs it
    scene_terrain_cache: dict[str, str | None] = {}
    for roi_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("ROIs")):
        roi, season = _season_roi(roi_dir.name)
        for s1d in sorted(roi_dir.glob("s1_*")):
            scene = s1d.name.split("_", 1)[1]
            s2d, lcd = roi_dir / f"s2_{scene}", roi_dir / f"lc_{scene}"
            if not s2d.is_dir() or not lcd.is_dir():
                continue
            cache_key = f"{roi}_{season}_{scene}"
            if cache_key not in scene_terrain_cache:
                first_lc = next(lcd.glob("*.tif"), None)
                if first_lc is None:
                    scene_terrain_cache[cache_key] = None
                else:
                    with rasterio.open(first_lc) as lc:
                        igbp = lc.read(1)
                    vals, cnts = np.unique(igbp, return_counts=True)
                    scene_terrain_cache[cache_key] = IGBP_TO_TERRAIN.get(int(vals[cnts.argmax()]))
            terrain = scene_terrain_cache[cache_key]
            if terrain is None:
                continue
            for s1f in sorted(s1d.glob("*.tif")):
                base = s1f.name.replace("_s1_", "_s2_")
                s2f = s2d / base
                if not s2f.exists():
                    continue
                patch = re.search(r"_p(\d+)\.tif$", s1f.name)
                yield {"source": "sen12ms", "season": season, "roi": roi, "scene": scene,
                       "patch": patch.group(1) if patch else "0", "terrain": terrain,
                       "s1_tif": s1f, "s2_tif": s2f}


# ----------------------------------------------------------------------------- #
#  Loaders: each record -> (sar_uint8 HxW, rgb_uint8 HxWx3) after preprocessing
# ----------------------------------------------------------------------------- #
def load_record(rec, sar_clip, s2_clip):
    if "sar_png" in rec:                                   # v_2 / SEN1-2: already 8-bit PNG
        sar = np.asarray(Image.open(rec["sar_png"]).convert("L"), dtype=np.uint8)
        rgb = np.asarray(Image.open(rec["eo_png"]).convert("RGB"), dtype=np.uint8)
    else:                                                  # SEN12MS GeoTIFF
        import rasterio
        with rasterio.open(rec["s1_tif"]) as s1:
            vv = s1.read(1).astype(np.float32)             # band 1 = VV (dB)
        with rasterio.open(rec["s2_tif"]) as s2:
            b2, b3, b4 = s2.read(2).astype(np.float32), s2.read(3).astype(np.float32), s2.read(4).astype(np.float32)
        sar = sar_db_to_uint8(vv, *sar_clip)
        rgb = s2_bands_to_rgb(b4, b3, b2, s2_clip)
    return sar, rgb


# ----------------------------------------------------------------------------- #
#  Reusable build steps (shared by this CLI and scripts/build_final_dataset.py)
# ----------------------------------------------------------------------------- #
def group_scenes(iters, dedup: bool = True):
    """Consume per-source record iterators -> {(source,season,roi,scene,terrain): [recs]}.
    Dedup by (roi,season,scene,patch) so the same physical patch is never counted twice."""
    scenes: dict[tuple, list] = defaultdict(list)
    seen: set = set()
    count = 0
    for it in iters:
        for rec in it:
            count += 1
            if count % 5000 == 0:
                print(f"  indexing... {count} patches scanned, {len(scenes)} scenes found")
            pid = (rec["roi"], rec["season"], rec["scene"], rec["patch"])
            if dedup and pid in seen:
                continue
            seen.add(pid)
            scenes[(rec["source"], rec["season"], rec["roi"], rec["scene"], rec["terrain"])].append(rec)
    print(f"  indexed {count} patches -> {len(scenes)} scenes ({len(seen)} unique)")
    return scenes


def _breadth_first_pick(scene_lists, limit, rng):
    """Round-robin across scenes: 1 patch from each scene per pass, then 2nd, ... until limit."""
    rng.shuffle(scene_lists)
    picked = []
    while scene_lists and (limit is None or len(picked) < limit):
        next_round = []
        for sl in scene_lists:
            if limit is not None and len(picked) >= limit:
                break
            picked.append(sl.pop())
            if sl:
                next_round.append(sl)
        scene_lists = next_round
    return picked


def sample_plan(scenes, k_per_scene: int, target_total: int | None, rng):
    """Balanced, breadth-FIRST sampling with overflow redistribution.

    First pass: equal per-terrain budget. Second pass: if any terrain exhausted its data before
    reaching its budget, redistribute the shortfall to terrains that still have patches left.
    This way data-rich terrains (urban, agri) compensate for scarce ones (water, forest) and
    the total stays close to target_total."""
    per_terrain: dict[str, list[list]] = defaultdict(list)
    for key, recs in scenes.items():
        terrain = key[4]
        if terrain is None:
            continue
        idx = rng.permutation(len(recs))[:k_per_scene]
        per_terrain[terrain].append([recs[i] for i in idx])

    if target_total is None:
        plan = []
        for terrain in sorted(per_terrain):
            plan.extend(_breadth_first_pick(
                [list(sl) for sl in per_terrain[terrain]], None, rng))
        rng.shuffle(plan)
        return plan

    n_terrains = max(len(per_terrain), 1)
    per_terr_target = target_total // n_terrains
    plan = []
    leftover_scenes: list[list] = []
    shortfall = 0
    for terrain in sorted(per_terrain):
        scene_lists = [list(sl) for sl in per_terrain[terrain]]
        picked = _breadth_first_pick(scene_lists, per_terr_target, rng)
        plan.extend(picked)
        if len(picked) < per_terr_target:
            shortfall += per_terr_target - len(picked)
        else:
            leftover_scenes.extend(scene_lists)

    if shortfall > 0 and leftover_scenes:
        leftover_scenes = [sl for sl in leftover_scenes if sl]  # drop emptied lists
        extra = _breadth_first_pick(leftover_scenes, shortfall, rng)
        plan.extend(extra)

    rng.shuffle(plan)
    return plan


def plan_report(plan):
    """Print + return the terrain/season/source breakdown and the distinct-scene set of a plan."""
    by_terr, by_season, by_src = defaultdict(int), defaultdict(int), defaultdict(int)
    scene_set = set()
    for r in plan:
        by_terr[r["terrain"]] += 1; by_season[r["season"]] += 1; by_src[r["source"]] += 1
        scene_set.add((r["source"], r["roi"], r["season"], r["scene"]))
    print(f"PLAN: {len(plan)} pairs from {len(scene_set)} scenes")
    print("  by terrain:", dict(by_terr))
    print("  by season :", dict(by_season))
    print("  by source :", dict(by_src))
    return by_terr, by_season, by_src, scene_set


def _process_one(args_tuple):
    """Worker: load one record, quality-filter, save PNGs. Returns (scene_tag, name) or None."""
    rec, out_str, sar_clip, s2_clip, quality_filter = args_tuple
    out = Path(out_str)
    sar, rgb = load_record(rec, sar_clip, s2_clip)
    if sar.shape[:2] != (256, 256):
        sar = np.asarray(Image.fromarray(sar).resize((256, 256)), dtype=np.uint8)
        rgb = np.asarray(Image.fromarray(rgb).resize((256, 256)), dtype=np.uint8)
    if quality_filter and (is_degenerate_sar(sar) or is_cloudy(rgb)):
        return None
    name = f"{rec['source']}_{rec['roi']}_{rec['season']}_{rec['scene']}_p{rec['patch']}.png"
    for sub, arr in [("s1", sar), ("s2", rgb)]:
        d = out / rec["terrain"] / sub
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(d / name)
    return f"{rec['terrain']}/{rec['source']}_{rec['roi']}_{rec['season']}_{rec['scene']}"


def write_pool(plan, out, sar_clip, s2_clip, quality_filter: bool = True, seed: int = 42,
               k_per_scene: int | None = None, n_workers: int = 0) -> dict:
    """Write a plan to <out>/<terrain>/{s1,s2}/<name>.png (1-ch SAR VV uint8 + RGB uint8) and a
    build_manifest.json. Returns the manifest. This is the un-split 'pool'; run make_split_dirs
    (strategy='scene') on it for a leak-free train/val/test partition.

    n_workers=0 -> auto (cpu_count). Set 1 to disable multiprocessing."""
    import multiprocessing as mp
    import os

    out = Path(out)
    by_terr, by_season, by_src, scene_set = plan_report(plan)
    manifest = {"counts": len(plan), "n_scenes": len(scene_set),
                "by_terrain": dict(by_terr), "by_season": dict(by_season),
                "by_source": dict(by_src), "sar_clip_db": list(sar_clip),
                "s2_clip": s2_clip, "k_per_scene": k_per_scene, "seed": seed, "scenes": []}

    out.mkdir(parents=True, exist_ok=True)
    clip = tuple(sar_clip)
    work = [(r, str(out), clip, s2_clip, quality_filter) for r in plan]
    total = len(work)

    if n_workers == 0:
        n_workers = min(os.cpu_count() or 4, 16)
    print(f"  writing {total} records with {n_workers} workers...")

    kept, skipped = 0, 0
    if n_workers <= 1:
        for i, w in enumerate(work):
            res = _process_one(w)
            if res is None:
                skipped += 1
            else:
                manifest["scenes"].append(res); kept += 1
            if (i + 1) % 500 == 0 or i + 1 == total:
                print(f"  writing... {i+1}/{total} processed, {kept} kept, {skipped} filtered")
    else:
        with mp.Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_process_one, work, chunksize=32)):
                if res is None:
                    skipped += 1
                else:
                    manifest["scenes"].append(res); kept += 1
                if (i + 1) % 500 == 0 or i + 1 == total:
                    print(f"  writing... {i+1}/{total} processed, {kept} kept, {skipped} filtered")

    manifest["counts"] = kept
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"WROTE {kept} pairs -> {out}  (manifest: {out/'build_manifest.json'})")
    return manifest


def make_iters(v2_root=None, sen12_root=None, sen12ms_root=None):
    """Build the per-source record iterators for whichever roots are provided."""
    iters = []
    if v2_root:      iters.append(iter_v2(Path(v2_root)))
    if sen12_root:   iters.append(iter_sen12(Path(sen12_root)))
    if sen12ms_root: iters.append(iter_sen12ms(Path(sen12ms_root)))
    if not iters:
        raise SystemExit("Provide at least one source root (v2 / sen12 / sen12ms)")
    return iters


# ----------------------------------------------------------------------------- #
#  Main build
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_root", default=None, help="Kaggle terrain-segregated set")
    ap.add_argument("--sen12_root", default=None, help="SEN1-2 root (season ROI folders)")
    ap.add_argument("--sen12ms_root", default=None, help="SEN12MS root (needs rasterio)")
    ap.add_argument("--out", required=True, help="unified dataset output root")
    ap.add_argument("--k_per_scene", type=int, default=10, help="max patches kept per scene (breadth lever)")
    ap.add_argument("--target_total", type=int, default=16000, help="approx total pairs to keep")
    ap.add_argument("--sar_clip", type=float, nargs=2, default=(-25.0, 0.0), help="SEN12MS VV dB clip range")
    ap.add_argument("--s2_clip", type=float, default=3000.0, help="SEN12MS S2 reflectance clip")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_quality_filter", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="report sampling plan; write nothing")
    ap.add_argument("--workers", type=int, default=0, help="parallel workers (0=auto, 1=serial)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    iters = make_iters(args.v2_root, args.sen12_root, args.sen12ms_root)
    scenes = group_scenes(iters)
    plan = sample_plan(scenes, args.k_per_scene, args.target_total, rng)
    if args.dry_run:
        plan_report(plan)
        print("dry-run: nothing written."); return
    write_pool(plan, args.out, args.sar_clip, args.s2_clip,
               quality_filter=not args.no_quality_filter, seed=args.seed,
               k_per_scene=args.k_per_scene, n_workers=args.workers)
    print("Next: make_split_dirs(strategy='scene') on this root for a leak-free split.")


if __name__ == "__main__":
    main()
