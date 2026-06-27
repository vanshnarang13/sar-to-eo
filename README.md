# SAR to EO Image Translation (GalaxEye Technical Assignment)

Translate a Sentinel 1 SAR (VV) patch into the matching Sentinel 2 optical (RGB) image. The task is
ill posed: SAR carries no colour or spectral information, so one SAR input is consistent with many
plausible optical outputs. The model is therefore ranked on perceptual metrics (LPIPS and FID), with
pixel metrics (SSIM and PSNR) reported as secondary diagnostics.

**Approach.** A Hybrid Conditional GAN that couples a CNN branch (local texture, Res2Net plus
Squeeze and Excitation residual blocks) with a Transformer branch (global context) and fuses them,
judged by a three scale spectral norm PatchGAN. It is trained with adversarial, L1, VGG perceptual
and terrain classification losses. The baseline uses a standard global attention ViT branch; two
controlled ablations are provided: one replaces the global attention with windowed Swin attention,
and one halves the L1 weight from 80 to 40. See `report.pdf` / `report.docx` for the full write up.

## Requirements

- Python 3.9 or newer (tested on 3.9 locally and 3.11 on the cloud GPU).
- All dependencies are pinned in `requirements.txt` (PyTorch 2.8, torchvision 0.23, lpips,
  torchmetrics, torch-fidelity, numpy, pillow, pyyaml, tqdm, wandb, matplotlib).
- Runs on a single GPU (CUDA), Apple Silicon (MPS), or CPU. Inference fits within 16 GB VRAM and
  needs no internet. The code is OS independent and runs on Windows, macOS and Linux.

## Environment setup

macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset structure

The final dataset combines two permitted sources: the Kaggle terrain segregated Sentinel 1 and 2
set (four terrains from SEN1-2) and a SEN12MS subset (adding forest, water and shrub, plus more
scenes and seasons). It holds 21,339 paired patches over seven terrains, split scene disjoint so
validation and test are unseen geographies.

```
data/final_dataset/{train,val,test}/<terrain>/s1/<name>.png   # SAR VV, single channel 256x256 8-bit
data/final_dataset/{train,val,test}/<terrain>/s2/<name>.png   # optical RGB 256x256, same filename
   terrain in {agri, barren, forest, grass, shrub, urban, water}
```

SAR inputs are dB scaled and min max normalised to [0, 255], which is exactly the inference contract
format. A flat, reduced, terrain mixed sample of the test split is included so a reviewer can run the
full inference and evaluation pipeline quickly without downloading the whole dataset:

```
sample_test/sar/<name>.png   # 420 SAR inputs across all terrains, flat
sample_test/eo/<name>.png    # matching ground-truth RGB, same filenames
```

The dataset build logic (including how SAR VV and RGB B4/B3/B2 are extracted from SEN12MS GeoTIFFs)
is in `scripts/build_dataset.py` and `scripts/build_final_dataset.py`.

## Training

Each model is one config. Training logs per epoch train and validation losses, saves `loss_curve.png`
and the raw `losses.csv` / `losses.json`, and tracks the run in Weights and Biases (it degrades to
offline or disabled with no account).

```bash
# baseline (global attention ViT branch)
python -m src.train --config configs/hybrid_cgan_vit.yaml

# ablation A (windowed Swin attention branch)
python -m src.train --config configs/hybrid_cgan_swin.yaml

# ablation B (Swin branch, L1 weight 40)
python -m src.train --config configs/hybrid_cgan_swin_l1_40.yaml
```

Cloud training used Modal on a single NVIDIA H100. Entry points are in `modal_app.py`.

## Inference (conforms to the assignment I/O contract)

```bash
python infer.py --input_dir <sar_png_dir> --output_dir <out_dir> --weights <checkpoint.pt>
```

Input: a directory of single channel 256x256 8-bit SAR PNGs. Output: a directory of 256x256 RGB PNGs
with identical filenames. Runs on a single GPU, MPS or CPU, and needs no internet.

Example on the included sample set:
```bash
python infer.py --input_dir sample_test/sar --output_dir outputs/preds --weights <checkpoint.pt>
```

## Evaluation

```bash
python eval.py --pred_dir <generated_rgb_dir> --gt_dir <ground_truth_rgb_dir>
```

Reports LPIPS, FID, SSIM and PSNR over every PNG present in both directories, matched by filename.
Pass `--no_fid` to skip FID when offline. End to end on the included sample set:

```bash
python infer.py --input_dir sample_test/sar --output_dir outputs/preds --weights <checkpoint.pt>
python eval.py  --pred_dir outputs/preds   --gt_dir sample_test/eo
```

## Model weights

Public download link for the final checkpoint: **[ADD LINK]** (Hugging Face Hub or Google Drive,
publicly accessible with no request-access step).

## Results

Validation metrics on the scene disjoint split (final epoch 150 for every model, so all are compared
at the same point in training; all four metrics from that same epoch):

| Model | LPIPS | FID | SSIM | PSNR |
|---|---|---|---|---|
| Baseline, ViT, L1 80 | 0.4211 | 87.18 | 0.5294 | 16.673 |
| Ablation A, Swin, L1 80 | 0.4225 | 92.15 | 0.5221 | 16.725 |
| Ablation B, Swin, L1 40 | 0.4205 | 90.32 | 0.5229 | 16.646 |

Test metrics on the complete test split (3,321 images across all terrains). A reviewer can reproduce
numbers on the included 420 image `sample_test` with the commands above; the full split is reported
here so the FID is reliable:

| Model | LPIPS | FID | SSIM | PSNR |
|---|---|---|---|---|
| Baseline, ViT, L1 80 | 0.4340 | 98.01 | 0.4306 | 15.576 |
| Ablation A, Swin, L1 80 | 0.4316 | 101.62 | 0.4256 | 15.726 |
| Ablation B, Swin, L1 40 | 0.4301 | 104.56 | 0.4315 | 15.679 |

Over the full split the test FID (98 to 105) is close to the validation FID above, confirming the
much higher values on the 420 image sample were small-sample bias; the per-image LPIPS, SSIM and PSNR
are directly comparable across splits.

Training and validation loss curves are saved per run at `outputs/runs/<run_name>/loss_curve.png`.

## Repository layout

```
configs/                  # full hyperparameter configs (baseline + 2 ablations)
src/data/dataset.py       # paired loader, scene-disjoint split builder, normalisation
src/models/hybrid_cgan.py # HybridGenerator (ViT or Swin branch) + multi-scale PatchGAN
src/models/swin_blocks.py # windowed / shifted-window Swin blocks (ablation)
src/losses.py             # GAN, L1, VGG perceptual, classification losses
src/metrics.py            # LPIPS / FID / SSIM / PSNR
src/train.py              # training loop (W&B, loss logging, checkpoints)
src/eval.py               # metric computation core
eval.py                   # evaluation entry point (--pred_dir / --gt_dir)
infer.py                  # contract-conforming inference
modal_app.py              # cloud-GPU training / evaluation entry points
scripts/                  # dataset build + EDA
sample_test/              # flat reduced test set (sar/ inputs + eo/ ground truth)
```

## Citations and references

This work uses Sentinel derived data; the Copernicus licence requires the following attribution:

> Contains modified Copernicus Sentinel data 2017 to 2018, processed by ESA.

Datasets:
- requiemonk. "Sentinel-1 and Sentinel-2 Image Pairs (Segregated by Terrain)." Kaggle.
  https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain
- Schmitt, Hughes, Zhu. "The SEN1-2 Dataset for Deep Learning in SAR-Optical Data Fusion." ISPRS
  Annals IV-1, 2018. CC BY 4.0. https://mediatum.ub.tum.de/1436631
- Schmitt, Hughes, Qiu, Zhu. "SEN12MS: A Curated Dataset of Georeferenced Multi-Spectral
  Sentinel-1/2 Imagery for Deep Learning and Data Fusion." ISPRS Annals IV-2/W7, 2019. CC BY.
  https://mediatum.ub.tum.de/1474000

Key method references: the SAR to optical translation survey (Wang et al. 2026), the Hybrid cGAN
paper (Coupling Global and Local Features), HVT-cGAN, Swin Transformer, the Multi-Scale SAR to
optical paper, Pix2Pix, Pix2PixHD, Res2Net, Squeeze and Excitation Networks, and Spectral
Normalization. Metrics: LPIPS and FID. Full list in the report.
