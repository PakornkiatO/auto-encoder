"""
End-to-end pipeline check on a single TRAINING sample.

Purpose (the "prove it works" check):
    Take one shot that was used during training, run its raw 800-point pressure
    curve through the COMPLETE pipeline -- autoencoder encoding -> MLP prediction
    -- and compare the predicted quality values against the actual measured ones.

A correctly implemented & trained model must reproduce a *training* sample's
quality closely. (This checks correctness of the pipeline, not generalization;
generalization is what the held-out test set measures.)

Pipeline traced for the chosen shot:
    raw curve (800) -> AE input scaler -> ConvAE.encode -> 16 features
                    -> MLP input scaler -> MLP -> inverse target scaler
                    -> predicted [W1,W2,W3,L1,L2,Total_weight,Part_weight]

Run
---
    python predict_one_sample.py            # auto-picks a training shot
    python predict_one_sample.py 7-3        # use a specific shot label
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import autoencoder_feature_extraction as ae
import quality_mlp as qm


def main():
    ae_cfg, mlp_cfg = ae.CFG, qm.CFG
    ae.set_reproducibility(ae_cfg.seed)
    device = ae.get_device()

    # ---- raw data ----
    X, labels, groups, n_time = ae.load_pressure_data(ae_cfg)
    label_to_row = {s: i for i, s in enumerate(labels)}

    # ---- (1) train the autoencoder, exactly as the AE pipeline does ----
    ae_tr, ae_va = ae.group_train_val_split(groups, ae_cfg.val_fraction, ae_cfg.seed)
    ae_scaler = StandardScaler()
    Xtr_std = ae_scaler.fit_transform(X[ae_tr]).astype(np.float32)
    Xall_std = ae_scaler.transform(X).astype(np.float32)
    pin = device.type == "cuda"
    trl = DataLoader(TensorDataset(torch.tensor(Xtr_std)), batch_size=ae_cfg.batch_size,
                     shuffle=True, pin_memory=pin)
    vll = DataLoader(TensorDataset(torch.tensor(Xall_std[ae_va])),
                     batch_size=ae_cfg.batch_size, pin_memory=pin)
    print("\n[1/3] Training ConvAE (encoder)...")
    conv, _, _ = ae.train_model(ae.ConvAE(n_time, ae_cfg.latent_dim), trl, vll,
                                ae_cfg, device, "conv")
    feats, _ = ae.extract_features(conv, torch.tensor(Xall_std), device)   # (n_shots, 16)

    # ---- (2) quality targets, aligned to the shot order ----
    q = pd.read_excel(mlp_cfg.quality_path).rename(columns={"K8025": "shot"})
    q["shot"] = q["shot"].astype(str)
    Y = q.set_index("shot").loc[labels, mlp_cfg.targets].to_numpy(dtype=np.float32)

    # ---- (3) train the MLP on the AE features ----
    m_tr, m_va, m_te = qm.group_split_3way(groups, mlp_cfg.val_fraction,
                                           mlp_cfg.test_fraction, mlp_cfg.seed)
    xs, ys = StandardScaler(), StandardScaler()
    Ftr = xs.fit_transform(feats[m_tr]); Fva = xs.transform(feats[m_va])
    Ytr = ys.fit_transform(Y[m_tr]);     Yva = ys.transform(Y[m_va])
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    print("\n[2/3] Training quality MLP (predictor)...")
    mlp = qm.QualityMLP(feats.shape[1], Y.shape[1], mlp_cfg.hidden, mlp_cfg.dropout)
    mlp, _ = qm.train_mlp(mlp, to_t(Ftr), to_t(Ytr), to_t(Fva), to_t(Yva), mlp_cfg, device)

    # ---- choose a TRAINING sample (must be in both AE-train and MLP-train) ----
    ae_train, mlp_train = set(ae_tr.tolist()), set(m_tr.tolist())
    if len(sys.argv) > 1:
        if sys.argv[1] not in label_to_row:
            raise SystemExit(f"Unknown shot label '{sys.argv[1]}'.")
        idx = label_to_row[sys.argv[1]]
    else:
        idx = next(i for i in m_tr.tolist() if i in ae_train)   # guaranteed training shot
    shot, grp = labels[idx], groups[idx] + 1
    print(f"\n[3/3] Chosen sample: shot {shot} (DOE group {grp})")
    print(f"      in AE training set : {idx in ae_train}")
    print(f"      in MLP training set: {idx in mlp_train}")

    # ---- run the ONE raw curve through the complete pipeline ----
    raw = X[idx][None, :]                                   # (1, 800) original units
    x_std = ae_scaler.transform(raw).astype(np.float32)     # AE input scaling
    with torch.no_grad():
        z = conv.encode(to_t(x_std).to(device)).cpu().numpy()      # encode -> 16 features
    assert np.allclose(z, feats[idx], atol=1e-4), "encoding path mismatch"   # cross-check
    f_std = xs.transform(z)                                 # MLP input scaling
    with torch.no_grad():
        pred_std = mlp(to_t(f_std).to(device)).cpu().numpy()
    pred = ys.inverse_transform(pred_std)[0]                # back to original units
    actual = Y[idx]

    # ---- report ----
    rep = pd.DataFrame({"target": mlp_cfg.targets, "actual": actual, "predicted": pred})
    rep["abs_error"] = (rep["actual"] - rep["predicted"]).abs()
    rep["pct_error"] = 100 * rep["abs_error"] / rep["actual"].abs()
    rep = rep.round({"actual": 4, "predicted": 4, "abs_error": 4, "pct_error": 3})

    print(f"\n=== Pipeline reproduction for training shot {shot} ===")
    print(rep.to_string(index=False))
    print(f"\nmean absolute % error: {rep['pct_error'].mean():.3f}%")
    print("(small error on a training sample => the full pipeline is wired and "
          "trained correctly)")


if __name__ == "__main__":
    main()
