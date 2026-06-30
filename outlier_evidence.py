"""
Evidence that shot 75-2's Total_weight was a data error, and that fixing it
solves the problem -- shown with the SAME measure (predicted vs actual) before
and after, for a fair comparison.

Figure (outputs/evidence/total_weight_outlier_evidence.png), two panels:
  (A) BEFORE fix: Total_weight predicted vs actual on the original (buggy) data
      where 75-2 = 17.26. The single bad point sits far off the y=x line
      (test R^2 ~ 0.05) -- "only one case separates from the rest".
  (B) AFTER fix: the identical predicted-vs-actual plot on the corrected data
      (75-2 = 14.26). Now 75-2 lies on the line with every other shot
      (test R^2 ~ 0.99) -- the problem is solved.

The decision to fix (rather than drop) Total_weight was based on the Pearson
correlation coefficient with Part_weight: r = 0.451 with the typo vs 0.965
without it -- a single point destroying an otherwise strong linear relationship,
i.e. the value IS predictable. (Printed to the console.)

Note: AE features depend only on the pressure curves, so they are unaffected by
the quality typo; we reuse the latest run's saved features and only vary targets.

Output: outputs/evidence/total_weight_outlier_evidence.png
"""

from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import autoencoder_feature_extraction as ae
import quality_mlp as qm

OUT_DIR = os.path.join(ae.BASE_DIR, "outputs", "evidence")
BUGGY_VALUE = 17.26          # the original (erroneous) Total_weight for shot 75-2


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def train_predict(X, Y, tr, va, te, device):
    """Train the quality MLP on targets Y and return test-set predictions."""
    torch.manual_seed(qm.CFG.seed)            # same init for a fair before/after
    xs, ys = StandardScaler(), StandardScaler()
    Xtr, Xva, Xte = xs.fit_transform(X[tr]), xs.transform(X[va]), xs.transform(X[te])
    Ytr, Yva = ys.fit_transform(Y[tr]), ys.transform(Y[va])
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    mlp = qm.QualityMLP(X.shape[1], Y.shape[1], qm.CFG.hidden, qm.CFG.dropout)
    mlp, _ = qm.train_mlp(mlp, to_t(Xtr), to_t(Ytr), to_t(Xva), to_t(Yva), qm.CFG, device)
    mlp.eval()
    with torch.no_grad():
        return ys.inverse_transform(mlp(to_t(Xte).to(device)).cpu().numpy())


def panel(ax, actual, pred, r2, title, te, out, color):
    ax.scatter(actual, pred, s=30, alpha=0.7)
    lo = min(actual.min(), pred.min()); hi = max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal y = x")
    if out in set(te.tolist()):
        j = list(te).index(out)
        ax.scatter([actual[j]], [pred[j]], s=160, facecolors="none",
                   edgecolors=color, linewidths=2, label="shot 75-2")
        ax.annotate("75-2", (actual[j], pred[j]),
                    textcoords="offset points", xytext=(8, -6), color=color)
    ax.set_title(title)
    ax.set_xlabel("actual Total_weight (g)"); ax.set_ylabel("predicted (g)")
    ax.legend(fontsize=8)


def main():
    ae.set_reproducibility(ae.CFG.seed)
    device = ae.get_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- features from the latest AE run (independent of the quality typo) ----
    run = max(glob.glob(os.path.join(ae.CFG.out_dir, "L*")), key=os.path.getmtime)
    feat = pd.read_csv(os.path.join(run, "features_conv.csv"))
    feat["shot"] = feat["shot"].astype(str)
    fcols = [c for c in feat.columns if c not in ("shot", "doe_group")]
    X = feat[fcols].to_numpy(np.float32)
    labels = feat["shot"].tolist()
    groups = feat["doe_group"].to_numpy() - 1

    # ---- targets: current (fixed) and a reconstructed BUGGY copy ----
    q = pd.read_excel(qm.CFG.quality_path).rename(columns={"K8025": "shot"})
    q["shot"] = q["shot"].astype(str)
    q = q.set_index("shot").loc[labels].reset_index()
    targets = qm.CFG.targets
    ti, pi = targets.index("Total_weight"), targets.index("Part_weight")
    Y = q[targets].to_numpy(np.float32)
    out = labels.index("75-2")
    Y_bug = Y.copy(); Y_bug[out, ti] = BUGGY_VALUE

    tr, va, te = qm.group_split_3way(groups, qm.CFG.val_fraction,
                                     qm.CFG.test_fraction, qm.CFG.seed)

    # ---- same measure (predicted vs actual) BEFORE and AFTER the fix ----
    pred_bug = train_predict(X, Y_bug, tr, va, te, device)
    pred_fix = train_predict(X, Y,     tr, va, te, device)
    a_bug, p_bug = Y_bug[te, ti], pred_bug[:, ti]
    a_fix, p_fix = Y[te, ti],     pred_fix[:, ti]
    r2_bug, r2_fix = r2_score(a_bug, p_bug), r2_score(a_fix, p_fix)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    panel(ax[0], a_bug, p_bug, r2_bug,
          f"(A) Before fix (original data)\n"
          f"Total_weight test R2 = {r2_bug:.2f} -- 75-2 off the line",
          te, out, "red")
    panel(ax[1], a_fix, p_fix, r2_fix,
          f"(B) After fix (corrected data)\n"
          f"Total_weight test R2 = {r2_fix:.2f} -- 75-2 on the line",
          te, out, "green")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "total_weight_outlier_evidence.png")
    plt.savefig(path, dpi=130); plt.close()

    # ---- detection rationale: correlation coefficient ----
    pw = Y[:, pi]
    r_fixed, r_typo = pearson(Y[:, ti], pw), pearson(Y_bug[:, ti], pw)
    print("\n--- Decision (Pearson correlation Total vs Part) ---")
    print("  r = cov(Total_weight, Part_weight) / (std_Total * std_Part)")
    print(f"  with original 75-2 typo: r = {r_typo:.4f}   (r^2 = {r_typo**2:.3f})")
    print(f"  corrected data:          r = {r_fixed:.4f}   (r^2 = {r_fixed**2:.3f})")
    print("  -> a single point breaks a strong linear relation: predictable, so FIX not drop.")
    print(f"\n--- Same-measure check (Total_weight predicted vs actual) ---")
    print(f"  before fix: test R2 = {r2_bug:.3f}")
    print(f"  after fix : test R2 = {r2_fix:.3f}")
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
