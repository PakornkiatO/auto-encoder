"""Save/load helpers for trained models + scalers, so a trained model can be
reloaded later (e.g. for inference) instead of retraining.

Layout written by the training scripts:
    outputs/<run_tag>/
        conv_ae.pt, dense_ae.pt        - autoencoder weights (state_dict)
        ae_scaler.joblib               - pressure-curve StandardScaler (fit on K8025 train)
        ae_meta.json                   - {input_len, latent_dim, models}
        quality_mlp/
            mlp.pt                      - MLP weights
            mlp_x_scaler.joblib         - feature scaler
            mlp_y_scaler.joblib         - target scaler
            mlp_meta.json               - {in_dim, out_dim, hidden, dropout, targets}

The model classes are imported lazily inside the load functions, so importing
this module never triggers a circular import (the training modules import this
one, not the other way around at load time).
"""

from __future__ import annotations

import os
import glob
import json

import torch
import joblib

AE_META, AE_SCALER = "ae_meta.json", "ae_scaler.joblib"
MLP_META = "mlp_meta.json"


# --------------------------------------------------------------------------- #
# Autoencoder
# --------------------------------------------------------------------------- #
def save_ae(run_dir, trained_models: dict, scaler, input_len, latent_dim):
    """Save each trained AE (state_dict), the input scaler, and metadata."""
    for name, m in trained_models.items():
        torch.save(m.state_dict(), os.path.join(run_dir, f"{name}_ae.pt"))
    joblib.dump(scaler, os.path.join(run_dir, AE_SCALER))
    with open(os.path.join(run_dir, AE_META), "w") as f:
        json.dump({"input_len": int(input_len), "latent_dim": int(latent_dim),
                   "models": list(trained_models)}, f, indent=2)


def ae_artifacts_exist(run_dir):
    return all(os.path.exists(os.path.join(run_dir, f))
               for f in (AE_META, AE_SCALER, "conv_ae.pt"))


def load_conv_ae(run_dir, device):
    """Return (ConvAE in eval mode on device, fitted scaler, meta)."""
    from autoencoder_feature_extraction import ConvAE  # lazy: avoids circular import
    if not ae_artifacts_exist(run_dir):
        raise SystemExit(f"No saved autoencoder in {run_dir}.\n"
                         f"Run autoencoder_feature_extraction.py first.")
    meta = json.load(open(os.path.join(run_dir, AE_META)))
    model = ConvAE(meta["input_len"], meta["latent_dim"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "conv_ae.pt"),
                                     map_location=device))
    model.to(device).eval()
    scaler = joblib.load(os.path.join(run_dir, AE_SCALER))
    return model, scaler, meta


# --------------------------------------------------------------------------- #
# Quality MLP
# --------------------------------------------------------------------------- #
def save_mlp(mlp_dir, model, x_scaler, y_scaler, in_dim, out_dim, hidden, dropout, targets):
    """Save the MLP (state_dict), its feature/target scalers, and metadata."""
    torch.save(model.state_dict(), os.path.join(mlp_dir, "mlp.pt"))
    joblib.dump(x_scaler, os.path.join(mlp_dir, "mlp_x_scaler.joblib"))
    joblib.dump(y_scaler, os.path.join(mlp_dir, "mlp_y_scaler.joblib"))
    with open(os.path.join(mlp_dir, MLP_META), "w") as f:
        json.dump({"in_dim": int(in_dim), "out_dim": int(out_dim),
                   "hidden": list(hidden), "dropout": float(dropout),
                   "targets": list(targets)}, f, indent=2)


def mlp_artifacts_exist(mlp_dir):
    return all(os.path.exists(os.path.join(mlp_dir, f))
               for f in (MLP_META, "mlp.pt", "mlp_x_scaler.joblib", "mlp_y_scaler.joblib"))


def load_mlp(mlp_dir, device):
    """Return (QualityMLP in eval mode on device, x_scaler, y_scaler, meta)."""
    from quality_mlp import QualityMLP  # lazy: avoids circular import
    if not mlp_artifacts_exist(mlp_dir):
        raise SystemExit(f"No saved MLP in {mlp_dir}.\n"
                         f"Run quality_mlp.py on that feature run first.")
    meta = json.load(open(os.path.join(mlp_dir, MLP_META)))
    model = QualityMLP(meta["in_dim"], meta["out_dim"], tuple(meta["hidden"]), meta["dropout"])
    model.load_state_dict(torch.load(os.path.join(mlp_dir, "mlp.pt"), map_location=device))
    model.to(device).eval()
    xs = joblib.load(os.path.join(mlp_dir, "mlp_x_scaler.joblib"))
    ys = joblib.load(os.path.join(mlp_dir, "mlp_y_scaler.joblib"))
    return model, xs, ys, meta


# --------------------------------------------------------------------------- #
# Run-folder discovery
# --------------------------------------------------------------------------- #
def latest_run_with(parent, check_fn):
    """Most recently modified run folder under `parent` satisfying check_fn(dir)."""
    runs = [d for d in glob.glob(os.path.join(parent, "*"))
            if os.path.isdir(d) and check_fn(d)]
    return max(runs, key=os.path.getmtime) if runs else None
