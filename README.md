# K8025 Pressure-Curve Feature Extraction

Undercomplete autoencoder feature extraction for K8025 injection-molding
**packing/cooling pressure curves**. Each shot is an 800-point pressure trace
(800 time steps @ 1000 Hz); the script compresses every curve into a **16-D
feature vector** using two autoencoders (a dense MLP and a 1D-conv) and compares
them.

## Project layout

```
intern/
├── autoencoder_feature_extraction.py   # train AEs, extract 16-D features per shot
├── quality_mlp.py                      # predict quality targets from AE features
├── predict_one_sample.py               # end-to-end check on one training shot
├── verify_ae.py                        # autoencoder health checks (pass/fail)
├── requirements.txt
├── README.md
├── data/                               # raw data — NOT in git, copy manually
│   ├── K8025_PackingCooling_Pressure-Data.csv   # 800 rows (time) x 450 cols (shots)
│   ├── K8025_Parameter.xlsx
│   └── K8025_weight-quality.xlsx
└── outputs/                            # generated results, one folder per run (gitignored)
    └── <run_tag>/                      # e.g. L16_lr1e-03_wd1e-05_bs32_<timestamp>
```

> **Note:** `data/` and `outputs/` are gitignored. If you clone via git, the raw
> data will **not** come with it — copy the `data/` folder over separately.

## Setup

Target machine: Windows + NVIDIA RTX 3070 Ti (CUDA). Uses conda.

```bat
conda create -n intern python=3.11
conda activate intern

:: 1) Install the CUDA build of PyTorch FIRST (matches the RTX 3070 Ti):
pip install torch --index-url https://download.pytorch.org/whl/cu124

:: 2) Install the remaining dependencies:
pip install -r requirements.txt
```

`pip install torch` without the index URL installs a **CPU-only** build, so do
step 1 before step 2. (On macOS/Linux the commands are the same; just use a
shell instead of `cmd`.)

## Run

```bat
conda activate intern
python autoencoder_feature_extraction.py
```

The first line of output reports the device:

```
Device: CUDA -> NVIDIA GeForce RTX 3070 Ti (8.0 GB)
```

If it prints `Device: CPU` instead, the CUDA PyTorch install didn't take —
re-check the install step and `nvidia-smi`. A full run takes well under a minute.

## Outputs

Written to `outputs/<run_tag>/`:

| File | Contents |
|------|----------|
| `features_dense.csv`, `features_conv.csv` | 450 shots × 16 latent features (labeled by `shot` and `doe_group`) |
| `recon_error_per_shot.csv` | per-shot reconstruction MSE for both models, with train/val split |
| `summary.txt` | final metrics comparison |
| `training_curves.png` | train/val loss for both models |
| `reconstructions_<model>.png` | example curve reconstructions |
| `latent_pca_<model>.png` | 2D PCA view of the 16-D latent space |

## Notes

- The script auto-detects the device (CUDA → Apple MPS → CPU) and runs
  reproducibly (fixed seeds + deterministic algorithms).
- Data orientation: rows = time steps, columns = shots; every 5 columns is one
  DOE setting (90 groups × 5 shots = 450). Train/val split is done at the **DOE
  group level** so no group spans both sets.
- Configuration (latent dim, epochs, batch size, etc.) lives in the `Config`
  dataclass near the top of the script.
