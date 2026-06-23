"""
Quality-prediction MLP for K8025 injection molding.

Maps the autoencoder's extracted pressure-curve features (16-D per shot) to the
measured part-quality targets:

    W1, W2, W3      - top / middle / bottom WIDTH of the part
    L1, L2          - long-side LENGTHS
    Total_weight    - total weight (part + runner/sprue)
    Part_weight     - weight of the part itself

A single multi-output MLP predicts all 7 targets from the 16 conv-AE features.

Feature source
--------------
The autoencoder writes one versioned folder per run under
    outputs/autoencoder_outputs/<run_tag>/features_conv.csv
This script lets you PICK which run's features to train on (interactive menu,
or pass the run folder name/path as the first CLI argument).

Methodology
-----------
- Group-level train/val/test split by DOE group (the 5 shots per group are
  near-identical replicates, so a random split would leak). Val is for early
  stopping; test is held out for the reported numbers.
- Inputs and targets are standardized on TRAIN only; predictions are mapped
  back to original units for reporting.
- Reported per target on the test set: R2 and RMSE (original units), compared
  against a mean-predictor baseline and a linear-regression baseline so we can
  see whether the MLP (and the AE features) actually help.

Outputs (written to <chosen run folder>/quality_mlp/):
    metrics.csv            - per-target R2 / RMSE for MLP vs baselines
    predictions_test.csv   - per test shot: actual + predicted for each target
    pred_vs_actual.png     - scatter (predicted vs actual) per target
    summary.txt            - headline metrics

Run
---
    python quality_mlp.py                 # interactive feature-folder menu
    python quality_mlp.py <run_tag>       # use a specific run folder
"""

from __future__ import annotations

import os
import sys
import glob
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Reuse the autoencoder script's reproducibility + device helpers (no duplication).
from autoencoder_feature_extraction import set_reproducibility, get_device, BASE_DIR


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    runs_dir: str = os.path.join(BASE_DIR, "outputs", "autoencoder_outputs")
    quality_path: str = os.path.join(BASE_DIR, "data", "K8025_weight-quality.xlsx")
    features_file: str = "features_conv.csv"   # which AE features to use as input
    targets: list = field(default_factory=lambda: [
        "W1", "W2", "W3", "L1", "L2", "Total_weight", "Part_weight"])

    shots_per_group: int = 5
    val_fraction: float = 0.15     # split at the DOE-group level
    test_fraction: float = 0.15

    hidden: tuple = (64, 32)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 2000
    patience: int = 150            # early-stopping patience on val loss
    seed: int = 42


CFG = Config()


# --------------------------------------------------------------------------- #
# Feature-folder selection
# --------------------------------------------------------------------------- #
def list_feature_runs(cfg: Config):
    """Return run folders with the chosen features file, oldest -> newest by mtime.

    Sorted by modification time (not name), since the run tag starts with the
    hyperparameters, so an alphabetical sort would not be chronological.
    """
    runs = [r for r in glob.glob(os.path.join(cfg.runs_dir, "*"))
            if os.path.isdir(r) and os.path.exists(os.path.join(r, cfg.features_file))]
    return sorted(runs, key=os.path.getmtime)


def select_feature_folder(cfg: Config, argv_choice: str | None = None) -> str:
    """Pick which AE run's features to train on (CLI arg, or interactive menu)."""
    runs = list_feature_runs(cfg)
    if not runs:
        raise SystemExit(
            f"No '{cfg.features_file}' found under {cfg.runs_dir}.\n"
            f"Run autoencoder_feature_extraction.py first.")

    # Non-interactive: a run folder name or path was passed on the command line.
    if argv_choice:
        for r in runs:
            if argv_choice in (r, os.path.basename(r)):
                return r
        raise SystemExit(f"Requested feature run '{argv_choice}' not found.\n"
                         f"Available: {[os.path.basename(r) for r in runs]}")

    print("\nAvailable feature runs (newest last):")
    for i, r in enumerate(runs, 1):
        print(f"  [{i}] {os.path.basename(r)}")
    while True:
        try:
            choice = input(f"Select a run [1-{len(runs)}] "
                           f"(Enter = {len(runs)}, the latest): ").strip()
        except EOFError:
            # No interactive stdin (e.g. piped / run under `conda run`): use latest.
            print(f"  no input available -> defaulting to latest: "
                  f"{os.path.basename(runs[-1])}")
            return runs[-1]
        if choice == "":
            return runs[-1]
        if choice.isdigit() and 1 <= int(choice) <= len(runs):
            return runs[int(choice) - 1]
        print("  invalid choice, try again.")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_features_and_targets(cfg: Config, run_dir: str):
    """Load AE features for the chosen run and align them with quality targets."""
    feat = pd.read_csv(os.path.join(run_dir, cfg.features_file))
    feat["shot"] = feat["shot"].astype(str)
    feature_cols = [c for c in feat.columns if c.startswith("f")]

    q = pd.read_excel(cfg.quality_path).rename(columns={"K8025": "shot"})
    q["shot"] = q["shot"].astype(str)

    df = feat.merge(q[["shot"] + cfg.targets], on="shot", how="inner")
    if len(df) != len(feat):
        raise SystemExit(f"Feature/target mismatch: {len(feat)} features vs "
                         f"{len(df)} matched on 'shot'.")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    Y = df[cfg.targets].to_numpy(dtype=np.float32)
    groups = df["doe_group"].to_numpy()
    print(f"Loaded {len(df)} shots | {X.shape[1]} features -> {Y.shape[1]} targets")
    return X, Y, groups, df["shot"].tolist(), feature_cols


def group_split_3way(groups, val_frac, test_frac, seed):
    """Train/val/test split at the DOE-group level (no group spans two sets)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rng.shuffle(ug)
    n = len(ug)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_g = set(ug[:n_test].tolist())
    val_g = set(ug[n_test:n_test + n_val].tolist())

    test_idx = np.array([i for i, g in enumerate(groups) if g in test_g])
    val_idx = np.array([i for i, g in enumerate(groups) if g in val_g])
    train_idx = np.array([i for i, g in enumerate(groups)
                          if g not in test_g and g not in val_g])
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class QualityMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, dropout):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(model, Xtr, Ytr, Xva, Yva, cfg, device):
    """Full-batch training (the dataset is tiny) with early stopping on val MSE."""
    model.to(device)
    Xtr, Ytr, Xva, Yva = (t.to(device) for t in (Xtr, Ytr, Xva, Yva))
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optim.zero_grad(set_to_none=True)
        loss = loss_fn(model(Xtr), Ytr)
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            va = loss_fn(model(Xva), Yva).item()
        history["train"].append(loss.item())
        history["val"].append(va)

        if va < best_val - 1e-7:
            best_val, bad = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 100 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d} | train {loss.item():.4f} | val {va:.4f} | best {best_val:.4f}")
        if bad >= cfg.patience:
            print(f"  early stopping at epoch {epoch} (best val {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def per_target_metrics(y_true, y_pred, targets):
    """Return per-target R2 and RMSE (original units)."""
    rows = []
    for j, t in enumerate(targets):
        r2 = r2_score(y_true[:, j], y_pred[:, j])
        rmse = float(np.sqrt(np.mean((y_true[:, j] - y_pred[:, j]) ** 2)))
        rows.append({"target": t, "r2": r2, "rmse": rmse})
    return pd.DataFrame(rows)


def plot_pred_vs_actual(y_true, y_pred, targets, r2_by_target, out_path):
    n = len(targets)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(4 * ncols, 3.6 * nrows))
    for j, t in enumerate(targets):
        ax = plt.subplot(nrows, ncols, j + 1)
        yt, yp = y_true[:, j], y_pred[:, j]
        ax.scatter(yt, yp, s=18, alpha=0.7)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)   # ideal y = x
        ax.set_title(f"{t}  (R2={r2_by_target[t]:.2f})")
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    cfg = CFG
    set_reproducibility(cfg.seed)

    argv_choice = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir = select_feature_folder(cfg, argv_choice)
    print(f"Using features from: {os.path.basename(run_dir)}")

    out_dir = os.path.join(run_dir, "quality_mlp")
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    # ----- data -----
    X, Y, groups, shots, feature_cols = load_features_and_targets(cfg, run_dir)
    tr, va, te = group_split_3way(groups, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    print(f"Split (by DOE group): train {len(tr)} | val {len(va)} | test {len(te)} shots")

    # ----- scale on train only -----
    xs, ys = StandardScaler(), StandardScaler()
    Xtr = xs.fit_transform(X[tr]); Xva = xs.transform(X[va]); Xte = xs.transform(X[te])
    Ytr = ys.fit_transform(Y[tr]); Yva = ys.transform(Y[va])

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    model = QualityMLP(X.shape[1], Y.shape[1], cfg.hidden, cfg.dropout)
    print("\n=== Training quality MLP ===")
    model, history = train_mlp(model, to_t(Xtr), to_t(Ytr), to_t(Xva), to_t(Yva), cfg, device)

    # ----- predict test set (back to original units) -----
    model.eval()
    with torch.no_grad():
        pred_te_std = model(to_t(Xte).to(device)).cpu().numpy()
    pred_te = ys.inverse_transform(pred_te_std)
    Yte = Y[te]

    mlp_m = per_target_metrics(Yte, pred_te, cfg.targets)

    # ----- baselines -----
    lin = LinearRegression().fit(Xtr, Y[tr])          # features -> raw targets
    lin_pred = lin.predict(Xte)
    lin_m = per_target_metrics(Yte, lin_pred, cfg.targets)

    mean_pred = np.tile(Y[tr].mean(axis=0), (len(te), 1))   # predict train mean
    mean_m = per_target_metrics(Yte, mean_pred, cfg.targets)

    # ----- combine metrics -----
    metrics = mlp_m.rename(columns={"r2": "mlp_r2", "rmse": "mlp_rmse"})
    metrics["linreg_r2"] = lin_m["r2"].values
    metrics["linreg_rmse"] = lin_m["rmse"].values
    metrics["mean_rmse"] = mean_m["rmse"].values
    metrics = metrics.round(4)
    metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    # ----- predictions csv -----
    pred_df = pd.DataFrame({"shot": [shots[i] for i in te],
                            "doe_group": groups[te]})
    for j, t in enumerate(cfg.targets):
        pred_df[f"{t}_actual"] = Yte[:, j]
        pred_df[f"{t}_pred"] = pred_te[:, j]
    pred_df.to_csv(os.path.join(out_dir, "predictions_test.csv"), index=False)

    # ----- plot -----
    r2_by_target = dict(zip(metrics["target"], metrics["mlp_r2"]))
    plot_pred_vs_actual(Yte, pred_te, cfg.targets, r2_by_target,
                        os.path.join(out_dir, "pred_vs_actual.png"))

    # ----- summary -----
    lines = ["Quality-prediction MLP - summary",
             "=" * 50,
             f"features: {os.path.basename(run_dir)}/{cfg.features_file}",
             f"inputs: {X.shape[1]} features -> {Y.shape[1]} targets",
             f"split (groups): train {len(tr)} / val {len(va)} / test {len(te)} shots",
             "",
             "per-target test metrics (R2 higher=better, RMSE in original units):",
             metrics.to_string(index=False),
             "",
             f"mean test R2  -  MLP: {metrics['mlp_r2'].mean():.3f}  |  "
             f"LinReg: {metrics['linreg_r2'].mean():.3f}"]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
