"""Figures 3 and 5 for the second revision, in the house style of make_all_figs.py.

Fig 3 = make_all_figs.py lines 90-109, with ONE change requested by the Editor:
        panel (b) gets its own y-axis and y-axis label (sharey=False).
Fig 5 = replaces make_all_figs.py lines 136-168. The Editor asked for an expanded
        y-axis because lines overlapped; the figure is now three panels (S1/S2/S3),
        each with its own 0.15-wide y-range, all ten models, values = Tables 4 and 6.
"""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
DATA = "data"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)
S = ["Reent", "AccCtrl", "Arith", "UnchkRet", "DoS", "BadRnd", "FrontR", "TimeMan"]

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5,
    "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "errorbar.capsize": 2, "pdf.fonttype": 42,
})
MODEL_COLORS = {"BR-trunk": "#1f77b4", "LAT": "#ff7f0e", "LAT-mask": "#2ca02c",
                "LGC": "#d62728", "WIDENNET": "#9467bd", "RF-BR": "#8c564b",
                "LGBM-BR": "#e377c2", "CC-ens": "#7f7f7f", "RAkEL": "#bcbd22",
                "ML-kNN": "#17becf"}

# ---------------------------------------------------------------- Fig 3 -------
f1c = [f"f1_{x}" for x in S]
s2 = pd.read_csv(f"{DATA}/stage2_baselines_results.csv")
s3 = pd.read_csv(f"{DATA}/stage3_deep_results.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=False)      # CHANGED: was sharey=True
for ax, (df, models, title) in zip(axes, [
        (s2, ["RF-BR", "LGBM-BR", "CC-ens", "RAkEL", "ML-kNN"], "(a) classical"),
        (s3, ["BR-trunk", "LAT", "LAT-mask", "LGC", "WIDENNET"], "(b) deep")]):
    cm = df.groupby("model")[f1c].mean()
    cs = df.groupby(["model", "seed"])[f1c].mean().groupby("model").std()
    x = np.arange(8); w = 0.16
    for k, m in enumerate(models):
        ax.bar(x + (k - 2) * w, cm.loc[m], w, yerr=cs.loc[m], capsize=1.8,
               label=m, color=MODEL_COLORS[m], edgecolor="none",
               ecolor="0.15", error_kw={"linewidth": 0.7})
    ax.set_xticks(x); ax.set_xticklabels(S, rotation=45, ha="right")
    ax.set_title(title, pad=24); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.0)                                                  # CHANGED: same range on both axes
    ax.set_ylabel("per-label F1 (view-mean)")                            # CHANGED: label on EACH panel
for ax in axes:                      # each panel lists its OWN models
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5,
              fontsize=6.5, columnspacing=0.8, handlelength=1.4)
fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=2.0); fig.savefig(f"{FIG}/s1_perlabel_combined.pdf")
fig.savefig(f"{FIG}/s1_perlabel_combined.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Fig 5 -------
d5 = pd.read_csv(f"{DATA}/fig5_macro_f1_all.csv")
models = list(d5[d5.split == "S1"].sort_values("macro_f1", ascending=False).model)  # S1-descending
MARK = {m: mk for m, mk in zip(models, ["o", "s", "^", "D", "v", "o", "s", "^", "D", "v"])}
panels = [("S1", "(a) S1 (in-distribution)", (0.70, 0.85)),
          ("S2", "(b) S2 (compiler-era shift)", (0.40, 0.55)),
          ("S3", "(c) S3 (temporal shift)", (0.45, 0.60))]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
for ax, (sp, title, yl) in zip(axes, panels):
    sub = d5[d5.split == sp].set_index("model")
    for i, m in enumerate(models):
        sd = sub.loc[m, "macro_f1_sd"]
        ax.errorbar(i, sub.loc[m, "macro_f1"], yerr=None if pd.isna(sd) else sd,
                    fmt=MARK[m], ms=5.5, color=MODEL_COLORS[m], ecolor="0.15",
                    elinewidth=0.7, capsize=2, label=m)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=60, ha="right")
    ax.set_ylim(*yl); ax.set_yticks(np.round(np.arange(yl[0], yl[1] + 1e-9, 0.03), 2))
    ax.set_ylabel(r"macro-F1 ($\mathcal{F}_{\mathrm{all}}$)"); ax.set_title(title)
    ax.grid(axis="y", color="0.9", lw=0.5); ax.grid(axis="x", visible=False)
fig.tight_layout(w_pad=1.2); fig.savefig(f"{FIG}/s4_shift_degradation.pdf")
fig.savefig(f"{FIG}/s4_shift_degradation.png", dpi=200)
plt.close(fig)
print("Figures 3 and 5 regenerated")
