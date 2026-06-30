# K8025 Pressure-Curve Feature Extraction and Quality Prediction

A two-stage pipeline for K8025 injection-molding **packing/cooling pressure
curves**. Each shot is an 800-point pressure trace (800 time steps @ 1000 Hz).
(1) An **undercomplete autoencoder** compresses every curve into a **16-D
feature vector** (a dense MLP and a 1D-conv autoencoder are trained and
compared). (2) A **multi-output MLP** predicts the measured part-quality targets
(three widths, two lengths, two weights) from those features.

## Project layout

```
intern/
├── autoencoder_feature_extraction.py   # train AEs, extract 16-D features per shot
├── quality_mlp.py                      # predict quality targets from AE features
├── model_io.py                         # save trained models + scalers per run
├── outlier_evidence.py                 # analysis: data-error evidence figure
├── requirements.txt
├── README.md
├── summation, summation.tex            # project write-up (plain + LaTeX report)
├── data/                               # raw data — NOT in git, copy manually
│   ├── K8025_experiment/               # measured: pressure CSV, parameters, quality
│   └── K8025_simulation/               # simulated: pressure + quality
└── outputs/                            # generated results, one folder per run (gitignored)
    └── <run_tag>/                      # features, plots, summary, saved AE (*.pt/*.joblib),
                                        #   and quality_mlp/ (MLP results + saved MLP)
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

:: 1) Train autoencoders + extract features (writes a new outputs/<run_tag>/):
python autoencoder_feature_extraction.py

:: 2) Predict quality from a run's features (interactive menu, or pass a run tag):
python quality_mlp.py
```

The first line of output reports the device:

```
Device: CUDA -> NVIDIA GeForce RTX 3070 Ti (8.0 GB)
```

If it prints `Device: CPU` instead, the CUDA PyTorch install didn't take —
re-check the install step and `nvidia-smi`. A full run takes well under a minute.

## Outputs

**Autoencoder** — written to `outputs/<run_tag>/`:

| File | Contents |
|------|----------|
| `features_dense.csv`, `features_conv.csv` | 450 shots × 16 latent features (labeled by `shot` and `doe_group`) |
| `recon_error_per_shot.csv` | per-shot reconstruction MSE for both models, with train/val split |
| `summary.txt` | final metrics comparison |
| `training_curves.png` | train/val loss for both models |
| `reconstructions_<model>.png` | example curve reconstructions |
| `latent_pca_<model>.png` | 2D PCA view of the 16-D latent space |
| `conv_ae.pt`, `dense_ae.pt`, `ae_scaler.joblib`, `ae_meta.json` | saved trained AE + input scaler + metadata |

**Quality MLP** — written to `outputs/<run_tag>/quality_mlp/`:

| File | Contents |
|------|----------|
| `metrics.csv` | per-target R² / RMSE: MLP vs linear-regression and mean baselines |
| `predictions_test.csv` | per test-shot actual + predicted, each target |
| `pred_vs_actual.png` | predicted vs actual scatter, per target |
| `training_curve.png` | MLP train/val loss |
| `mlp.pt`, `mlp_x_scaler.joblib`, `mlp_y_scaler.joblib`, `mlp_meta.json` | saved trained MLP + scalers + metadata |

A top-level `outputs/runs_summary.csv` aggregates one row per AE run for comparison.

## Notes

- The script auto-detects the device (CUDA → Apple MPS → CPU) and runs
  reproducibly (fixed seeds + deterministic algorithms).
- Data orientation: rows = time steps, columns = shots; every 5 columns is one
  DOE setting (90 groups × 5 shots = 450). Train/val split is done at the **DOE
  group level** so no group spans both sets.
- Configuration (latent dim, epochs, batch size, etc.) lives in the `Config`
  dataclass near the top of the script.
