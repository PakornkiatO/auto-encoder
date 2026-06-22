"""
Undercomplete Autoencoder feature extraction for K8025 injection-molding
packing/cooling pressure curves.

Data
----
K8025_PackingCooling_Pressure-Data.csv : 800 rows x 450 columns
    - rows    = time steps sampled at 1000 Hz (1 ms each) -> 800 ms window
    - columns = shots (one injection-molding cycle each)
    - every 5 columns the machine setup changes -> 90 DOE groups x 5 shots = 450

Each shot is therefore one 800-point pressure curve and is treated as a single
sample. An *undercomplete* autoencoder (bottleneck dimension < input dimension,
here 16 << 800) is trained to reconstruct the curves; the bottleneck activations
are the extracted features.

Two architectures are trained and compared:
    1. DenseAE  - fully-connected MLP autoencoder ("classic" undercomplete AE)
    2. ConvAE   - 1D-convolutional autoencoder over the time axis

Target machine
--------------
Windows remote desktop: NVIDIA RTX 3070 Ti (CUDA, 8 GB VRAM),
Ryzen 9 5900X, 64 GB RAM.  The script auto-detects CUDA and uses
mixed precision (AMP) on the GPU, with deterministic/reproducible
settings, and falls back cleanly to MPS/CPU otherwise.

Requirements
------------
    conda create -n intern python=3.11
    conda activate intern
    # CUDA build of PyTorch for the RTX 3070 Ti, e.g.:
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install numpy pandas scikit-learn matplotlib openpyxl

Run
---
    python autoencoder_feature_extraction.py

Outputs (written to <script_dir>/outputs/autoencoder_outputs/):
    features_dense.csv / features_conv.csv  - (450 x 16) latent features per shot
    recon_error_per_shot.csv                - reconstruction MSE per shot per model
    training_curves.png                     - train/val loss for both models
    reconstructions_<model>.png             - example curve reconstructions
    latent_pca_<model>.png                  - 2D PCA of the 16-D latent space
    summary.txt                             - final metrics comparison
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless backend, save figures to disk
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Resolve paths relative to this file so the script runs from any working dir
# (important on the Windows remote desktop).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    data_path: str = os.path.join(
        BASE_DIR, "data", "K8025_PackingCooling_Pressure-Data.csv"
    )
    out_dir: str = os.path.join(BASE_DIR, "outputs", "autoencoder_outputs")

    shots_per_group: int = 5     # every 5 columns the machine setup changes
    latent_dim: int = 16         # bottleneck size (undercomplete: 16 << 800)

    val_fraction: float = 0.20   # split done at the DOE-group level (no leakage)
    batch_size: int = 32
    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 40           # early-stopping patience on val loss
    seed: int = 42

    use_amp: bool = True         # mixed precision when running on CUDA
    num_workers: int = 0         # 0 is safest on Windows (avoids spawn issues)


CFG = Config()


def set_reproducibility(seed: int) -> None:
    """Make CPU/CUDA runs reproducible. Call once, before device/model setup.

    Trades the cuDNN autotuner's small speedup for deterministic results, which
    is the right call here (the whole run takes seconds).
    """
    # Must be set before the first cuBLAS call for deterministic matmuls on CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                       # seeds CPU and all CUDA devices
    torch.backends.cudnn.benchmark = False         # no autotuning -> deterministic
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    """Detect compute device (pure — no global side effects): CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        print(f"Device: CUDA -> {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Device: Apple MPS")
        return torch.device("mps")
    print("Device: CPU")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_pressure_data(cfg: Config):
    """Return X (n_shots, n_timesteps) and a list of shot labels 'group-shot'."""
    # No header in the file; 800 rows (time) x 450 cols (shots).
    df = pd.read_csv(cfg.data_path, header=None)
    n_time, n_shots = df.shape
    print(f"Loaded raw data: {n_time} time steps x {n_shots} shots")

    # transpose -> each row is one shot (sample), each column a time step (feature)
    X = df.to_numpy(dtype=np.float32).T  # (n_shots, n_time)

    labels = [
        f"{i // cfg.shots_per_group + 1}-{i % cfg.shots_per_group + 1}"
        for i in range(n_shots)
    ]
    groups = np.array([i // cfg.shots_per_group for i in range(n_shots)])
    return X, labels, groups, n_time


def group_train_val_split(groups: np.ndarray, val_fraction: float, seed: int):
    """Split sample indices by DOE group so no group spans train and val."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n_val = max(1, int(round(len(unique_groups) * val_fraction)))
    val_groups = set(unique_groups[:n_val].tolist())

    val_idx = np.array([i for i, g in enumerate(groups) if g in val_groups])
    train_idx = np.array([i for i, g in enumerate(groups) if g not in val_groups])
    return train_idx, val_idx


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class DenseAE(nn.Module):
    """Fully-connected undercomplete autoencoder: 800 -> 256 -> 64 -> 16 -> ... -> 800."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),          # bottleneck (linear)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, input_dim),          # linear output (standardized space)
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class ConvAE(nn.Module):
    """1D-convolutional undercomplete autoencoder over the time axis (input length 800)."""

    def __init__(self, input_len: int, latent_dim: int):
        super().__init__()
        self.input_len = input_len
        # default for input_len=800: 800 -> 400 -> 200 -> 100 -> 50, channels 1->16->32->64->64
        self.enc_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
        )
        # Infer the flattened conv-output shape from a dummy forward pass so the
        # bottleneck adapts to any input length / encoder depth, instead of
        # assuming input_len is divisible by 2^(num stride-2 layers).
        with torch.no_grad():
            enc_out = self.enc_conv(torch.zeros(1, 1, input_len))
        self.conv_channels = enc_out.shape[1]    # 64 by default
        self.conv_out_len = enc_out.shape[2]      # 50 for input_len=800
        self.flat_feat = self.conv_channels * self.conv_out_len
        self.enc_fc = nn.Linear(self.flat_feat, latent_dim)   # bottleneck

        self.dec_fc = nn.Linear(latent_dim, self.flat_feat)
        # mirror the encoder back up; final length is fixed to input_len in forward()
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=8, stride=2, padding=3), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=2, padding=3), nn.ReLU(),
            nn.ConvTranspose1d(16, 16, kernel_size=8, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=7, padding=3),        # -> (1, L); fixed to input_len in forward()
        )

    def encode(self, x):
        # x: (N, input_len) -> add channel dim
        h = self.enc_conv(x.unsqueeze(1))
        h = h.flatten(1)
        return self.enc_fc(h)

    def forward(self, x):
        z = self.encode(x)
        h = self.dec_fc(z).view(-1, self.conv_channels, self.conv_out_len)
        out = self.dec_conv(h)                    # (N, 1, L)
        # Guarantee the output length matches the input (transpose-conv arithmetic
        # only lands exactly on input_len for nice sizes like 800); a no-op there.
        if out.shape[-1] != self.input_len:
            out = F.interpolate(out, size=self.input_len, mode="linear", align_corners=False)
        out = out.squeeze(1)                      # (N, input_len)
        return out, z


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(model, train_loader, val_loader, cfg, device, name):
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    amp_on = cfg.use_amp and device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=amp_on)
    # Async host->device copies are only safe from pinned memory (CUDA only).
    # On MPS the host tensor is unpinned, so non_blocking=True races and reads
    # garbage -> NaN. Keep it True only on CUDA.
    non_blocking = device.type == "cuda"

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=non_blocking)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp_on):
                recon, _ = model(xb)
                loss = loss_fn(recon, xb)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optim)
            scaler_amp.update()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device, non_blocking=non_blocking)
                with torch.autocast(device_type="cuda", enabled=amp_on):
                    recon, _ = model(xb)
                    loss = loss_fn(recon, xb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(val_loader.dataset)

        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"[{name}] epoch {epoch:4d} | train {tr_loss:.5f} | val {va_loss:.5f} | best {best_val:.5f}")

        if bad_epochs >= cfg.patience:
            print(f"[{name}] early stopping at epoch {epoch} (best val {best_val:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val


# --------------------------------------------------------------------------- #
# Feature extraction + evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(model, X_tensor, device, batch=64):
    model.eval()
    feats, recons = [], []
    for i in range(0, X_tensor.size(0), batch):
        xb = X_tensor[i:i + batch].to(device)
        recon, z = model(xb)
        feats.append(z.float().cpu().numpy())
        recons.append(recon.float().cpu().numpy())
    return np.concatenate(feats), np.concatenate(recons)


def per_shot_mse(X_true, X_recon):
    return np.mean((X_true - X_recon) ** 2, axis=1)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_training_curves(histories, out_path):
    plt.figure(figsize=(8, 5))
    for name, h in histories.items():
        plt.plot(h["train"], label=f"{name} train")
        plt.plot(h["val"], "--", label=f"{name} val")
    plt.xlabel("epoch")
    plt.ylabel("MSE (standardized space)")
    plt.yscale("log")
    plt.title("Autoencoder training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_reconstructions(X_orig, X_recon, labels, out_path, n=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_orig), size=min(n, len(X_orig)), replace=False)
    rows = (len(idx) + 2) // 3
    plt.figure(figsize=(13, 3 * rows))
    for k, i in enumerate(idx):
        plt.subplot(rows, 3, k + 1)
        plt.plot(X_orig[i], label="original", lw=1.2)
        plt.plot(X_recon[i], label="reconstruction", lw=1.2, alpha=0.8)
        plt.title(f"shot {labels[i]}")
        plt.xlabel("time (ms)")
        if k == 0:
            plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_latent_pca(features, groups, out_path, title):
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(features)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(z2[:, 0], z2[:, 1], c=groups, cmap="viridis", s=25)
    plt.colorbar(sc, label="DOE group index")
    var = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    cfg = CFG
    set_reproducibility(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = get_device()

    # ----- data -----
    X, labels, groups, n_time = load_pressure_data(cfg)
    train_idx, val_idx = group_train_val_split(groups, cfg.val_fraction, cfg.seed)
    print(f"Train shots: {len(train_idx)} | Val shots: {len(val_idx)} "
          f"(split by DOE group, no leakage)")

    # standardize per time step using TRAIN statistics only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_all = scaler.transform(X)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_all[val_idx], dtype=torch.float32)
    X_all_t = torch.tensor(X_all, dtype=torch.float32)

    pin = device.type == "cuda"
    train_loader = DataLoader(TensorDataset(X_train_t), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers, pin_memory=pin)
    val_loader = DataLoader(TensorDataset(X_val_t), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)

    models = {
        "dense": DenseAE(n_time, cfg.latent_dim),
        "conv": ConvAE(n_time, cfg.latent_dim),
    }

    histories, results = {}, {}
    val_set = set(val_idx.tolist())
    recon_errors = pd.DataFrame({"shot": labels, "doe_group": groups + 1,
                                 "split": ["val" if i in val_set else "train"
                                           for i in range(len(labels))]})

    for name, model in models.items():
        print(f"\n=== Training {name} autoencoder ===")
        model, hist, best_val = train_model(model, train_loader, val_loader, cfg, device, name)
        histories[name] = hist

        feats, recons = extract_features(model, X_all_t, device)

        # save latent features (in latent space) with shot labels
        feat_df = pd.DataFrame(feats, columns=[f"f{j:02d}" for j in range(cfg.latent_dim)])
        feat_df.insert(0, "doe_group", groups + 1)
        feat_df.insert(0, "shot", labels)
        feat_df.to_csv(os.path.join(cfg.out_dir, f"features_{name}.csv"), index=False)

        # reconstruction error in ORIGINAL units (inverse-transform)
        recons_orig = scaler.inverse_transform(recons)
        mse = per_shot_mse(X, recons_orig)
        recon_errors[f"mse_{name}"] = mse

        # Full-precision recompute of the val MSE from the restored best model,
        # as an integrity check against the (possibly AMP) train-time best_val.
        val_mse_std = per_shot_mse(X_all[val_idx], recons[val_idx]).mean()
        results[name] = {"best_val_std": best_val,
                         "val_mse_std": float(val_mse_std),
                         "overall_mse_orig": float(mse.mean())}

        plot_reconstructions(X, recons_orig, labels,
                             os.path.join(cfg.out_dir, f"reconstructions_{name}.png"))
        plot_latent_pca(feats, groups,
                        os.path.join(cfg.out_dir, f"latent_pca_{name}.png"),
                        f"{name} AE - 16-D latent space (PCA to 2D)")

    plot_training_curves(histories, os.path.join(cfg.out_dir, "training_curves.png"))
    recon_errors.to_csv(os.path.join(cfg.out_dir, "recon_error_per_shot.csv"), index=False)

    # ----- summary -----
    lines = ["Undercomplete Autoencoder feature extraction - summary",
             "=" * 55,
             f"input dim (time steps): {n_time}",
             f"samples (shots): {len(labels)}  |  DOE groups: {len(np.unique(groups))}",
             f"latent (feature) dim: {cfg.latent_dim}",
             f"train/val shots: {len(train_idx)}/{len(val_idx)} (group-level split)",
             ""]
    for name in models:
        r = results[name]
        lines.append(f"[{name}] best val MSE (std, train-time): {r['best_val_std']:.5f}  | "
                     f"recompute (full precision): {r['val_mse_std']:.5f}  | "
                     f"overall recon MSE (orig units): {r['overall_mse_orig']:.6f}")
    best_model = min(results, key=lambda n: results[n]["best_val_std"])
    lines.append("")
    lines.append(f"Lowest validation reconstruction error: '{best_model}' AE")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(cfg.out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    print(f"\nAll outputs written to {cfg.out_dir}")


if __name__ == "__main__":
    main()
