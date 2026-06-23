"""
Health checks for the K8025 autoencoder pipeline.

Answers two questions:
  (A) Is the code mechanically correct?  (shapes, NaNs, leakage, reproducibility)
  (B) Is the model actually learning?     (overfit test, beats mean + PCA baselines)

Each check prints [PASS]/[FAIL]; the script exits non-zero if any check fails,
so it doubles as a quick regression test after editing the AE.

Run
---
    python verify_ae.py
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from autoencoder_feature_extraction import (
    CFG, set_reproducibility, get_device, load_pressure_data,
    group_train_val_split, DenseAE, ConvAE, train_model,
)


# --------------------------------------------------------------------------- #
# Tiny check runner
# --------------------------------------------------------------------------- #
_RESULTS = []


def check(name, fn):
    """Run one check; record PASS/FAIL with a short message."""
    try:
        msg = fn() or ""
        _RESULTS.append((True, name, msg))
        print(f"[PASS] {name}  {msg}")
    except AssertionError as e:
        _RESULTS.append((False, name, str(e)))
        print(f"[FAIL] {name}  -> {e}")
    except Exception as e:  # unexpected error is also a failure
        _RESULTS.append((False, name, f"ERROR: {e!r}"))
        print(f"[FAIL] {name}  -> ERROR: {e!r}")


# --------------------------------------------------------------------------- #
# Shared fixtures (loaded once)
# --------------------------------------------------------------------------- #
def build_fixtures():
    set_reproducibility(CFG.seed)
    device = get_device()
    X, labels, groups, n_time = load_pressure_data(CFG)
    tr, va = group_train_val_split(groups, CFG.val_fraction, CFG.seed)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr]).astype(np.float32)
    Xva = scaler.transform(X[va]).astype(np.float32)
    return dict(device=device, X=X, groups=groups, n_time=n_time,
                tr=tr, va=va, Xtr=Xtr, Xva=Xva)


# --------------------------------------------------------------------------- #
# (A) Mechanical-correctness checks
# --------------------------------------------------------------------------- #
def check_shapes_and_nans(fx):
    n = fx["n_time"]
    x = torch.randn(5, n)
    for name, net in [("DenseAE", DenseAE(n, 16)), ("ConvAE", ConvAE(n, 16))]:
        recon, z = net(x)
        assert recon.shape == x.shape, f"{name} recon {tuple(recon.shape)} != input {tuple(x.shape)}"
        assert z.shape == (5, 16), f"{name} latent {tuple(z.shape)} != (5, 16)"
        assert torch.isfinite(recon).all(), f"{name} recon has NaN/Inf"
        assert torch.isfinite(z).all(), f"{name} latent has NaN/Inf"
    return "Dense & Conv: shapes match, no NaN"


def check_conv_arbitrary_length(_fx):
    # the ConvAE must adapt to lengths not divisible by 16 (the old bug)
    for L in (800, 813, 257):
        net = ConvAE(L, 16)
        out, z = net(torch.randn(3, L))
        assert out.shape == (3, L), f"L={L}: out {tuple(out.shape)} != (3, {L})"
        assert z.shape == (3, 16)
    return "ConvAE handles L in {800, 813, 257}"


def check_no_leakage(fx):
    tr_groups = set(fx["groups"][fx["tr"]].tolist())
    va_groups = set(fx["groups"][fx["va"]].tolist())
    overlap = tr_groups & va_groups
    assert not overlap, f"train/val share DOE groups: {sorted(overlap)[:5]}"
    return f"{len(tr_groups)} train / {len(va_groups)} val groups, disjoint"


def check_reproducible_init(fx):
    n = fx["n_time"]
    set_reproducibility(CFG.seed)
    a = ConvAE(n, 16)
    set_reproducibility(CFG.seed)
    b = ConvAE(n, 16)
    for (k, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(pa, pb), f"param {k} differs across identical seeds"
    return "same seed -> identical weights"


# --------------------------------------------------------------------------- #
# (B) Does-it-learn checks
# --------------------------------------------------------------------------- #
def check_overfit_tiny_batch(fx):
    """A correct AE must drive a tiny batch's loss near zero."""
    device = fx["device"]
    xb = torch.tensor(fx["Xtr"][:8], dtype=torch.float32).to(device)
    net = ConvAE(fx["n_time"], 16).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    init = final = None
    for i in range(1000):
        opt.zero_grad(set_to_none=True)
        recon, _ = net(xb)
        loss = ((recon - xb) ** 2).mean()
        loss.backward()
        opt.step()
        if i == 0:
            init = loss.item()
        final = loss.item()
    assert final < 0.05, f"could not overfit 8 samples (final MSE {final:.4f})"
    assert final < 0.2 * init, f"loss barely moved ({init:.3f} -> {final:.4f})"
    return f"8-sample MSE {init:.3f} -> {final:.4g}"


def _train_quick_conv(fx):
    """Train a ConvAE briefly on the real split; return standardized val MSE."""
    quick = replace(CFG, epochs=200, patience=40)
    Xtr_t = torch.tensor(fx["Xtr"], dtype=torch.float32)
    Xva_t = torch.tensor(fx["Xva"], dtype=torch.float32)
    pin = fx["device"].type == "cuda"
    trl = DataLoader(TensorDataset(Xtr_t), batch_size=quick.batch_size,
                     shuffle=True, pin_memory=pin)
    val = DataLoader(TensorDataset(Xva_t), batch_size=quick.batch_size, pin_memory=pin)
    _, _, best_val = train_model(ConvAE(fx["n_time"], 16), trl, val, quick, fx["device"], "verify")
    return best_val


def check_beats_mean_baseline(fx):
    """In standardized space, predicting the mean gives MSE ~= 1.0."""
    ae_val = fx.setdefault("ae_val", _train_quick_conv(fx))
    assert ae_val < 0.5, f"AE val MSE {ae_val:.3f} not clearly below mean baseline (~1.0)"
    return f"AE val MSE {ae_val:.3f} << 1.0 (explains ~{(1-ae_val)*100:.0f}% of variance)"


def check_vs_pca_baseline(fx):
    """A 16-D AE should be in the same ballpark as PCA(16) reconstruction."""
    ae_val = fx.setdefault("ae_val", _train_quick_conv(fx))
    p = PCA(n_components=16).fit(fx["Xtr"])
    rec = p.inverse_transform(p.transform(fx["Xva"]))
    pca_val = float(np.mean((rec - fx["Xva"]) ** 2))
    assert ae_val <= pca_val * 2.0, f"AE ({ae_val:.3f}) much worse than PCA ({pca_val:.3f})"
    return f"AE {ae_val:.3f} vs PCA(16) {pca_val:.3f} (ratio {ae_val/pca_val:.2f})"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=== Autoencoder verification ===")
    fx = build_fixtures()

    print("\n-- (A) mechanical correctness --")
    check("shapes & no NaN", lambda: check_shapes_and_nans(fx))
    check("ConvAE arbitrary length", lambda: check_conv_arbitrary_length(fx))
    check("no train/val leakage", lambda: check_no_leakage(fx))
    check("reproducible init", lambda: check_reproducible_init(fx))

    print("\n-- (B) does it actually learn --")
    check("overfit tiny batch", lambda: check_overfit_tiny_batch(fx))
    check("beats mean baseline", lambda: check_beats_mean_baseline(fx))
    check("competitive with PCA", lambda: check_vs_pca_baseline(fx))

    n_fail = sum(1 for ok, _, _ in _RESULTS if not ok)
    total = len(_RESULTS)
    print(f"\n=== {total - n_fail}/{total} checks passed ===")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
