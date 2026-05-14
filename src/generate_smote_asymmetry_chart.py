"""
Generate Article-2 SMOTE-asymmetry bar chart.

Two-panel bar chart (ULB | BAF) comparing FT-Transformer vs CatBoost PR-AUC
across the seven imbalance strategies. The visual story: on BAF, FT-Transformer
collapses 41% under SMOTE while CatBoost only drops 14% --- the architecture-
specific fragility that motivates the "simple trees suffice" argument.

Values are taken directly from the thesis Section 4 robustness tables
(tab:transformer_robustness_ulb and tab:transformer_robustness_baf).

Usage:
    cd src/
    python generate_smote_asymmetry_chart.py

Writes to: ../Article 2/figs/smote_asymmetry.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "Article 2" / "figs" / "smote_asymmetry.pdf"

# PR-AUC values from thesis robustness tables
PRAUC = {
    "ULB": {
        "FT-Trans.": [0.856, 0.650, 0.836, 0.845, 0.845, 0.772, 0.783],
        "CatBoost":  [0.885, 0.617, 0.879, 0.857, 0.857, 0.846, 0.875],
    },
    "BAF":  {
        "FT-Trans.": [0.180, 0.159, 0.167, 0.106, 0.106, 0.124, 0.175],
        "CatBoost":  [0.180, 0.178, 0.185, 0.154, 0.154, 0.165, 0.184],
    },
}

STRATEGIES = ["None", "RUS", "ROS", "SMOTE", "SM+T", "SMENN", "Wts"]
COLORS = {"FT-Trans.": "#ff7f0e", "CatBoost": "#d62728"}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for ax, dataset in zip(axes, ["ULB", "BAF"]):
        x = np.arange(len(STRATEGIES))
        width = 0.36

        bars_ft = ax.bar(x - width / 2, PRAUC[dataset]["FT-Trans."], width,
                         label="FT-Transformer",
                         color=COLORS["FT-Trans."], edgecolor="black", linewidth=0.7)
        bars_cb = ax.bar(x + width / 2, PRAUC[dataset]["CatBoost"], width,
                         label="CatBoost",
                         color=COLORS["CatBoost"], edgecolor="black", linewidth=0.7)

        # Highlight SMOTE column (index 3) and SM+Tomek (index 4)
        for idx in (3, 4):
            bars_ft[idx].set_hatch("//")
            bars_cb[idx].set_hatch("//")

        # Annotate the BAF asymmetry directly on the bars
        if dataset == "BAF":
            # FT-T baseline (None) and FT-T SMOTE drop annotation
            baseline = PRAUC["BAF"]["FT-Trans."][0]
            smote = PRAUC["BAF"]["FT-Trans."][3]
            drop_ft = (baseline - smote) / baseline * 100
            drop_cb = (PRAUC["BAF"]["CatBoost"][0] - PRAUC["BAF"]["CatBoost"][3]) / PRAUC["BAF"]["CatBoost"][0] * 100
            ax.annotate(
                f"FT-T: $-{drop_ft:.0f}\\%$\nCatBoost: $-{drop_cb:.0f}\\%$",
                xy=(3, smote + 0.005),
                xytext=(4.6, 0.16),
                fontsize=9.5,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGIES, fontsize=9.5, rotation=0)
        ax.set_xlabel("Imbalance strategy", fontsize=11)
        ax.set_ylabel("PR-AUC", fontsize=11)
        ax.set_title(f"{dataset}", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        if dataset == "ULB":
            ax.set_ylim([0.55, 0.95])
        else:
            ax.set_ylim([0.0, 0.22])

        ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

    fig.suptitle(
        "FT-Transformer vs.\\ CatBoost under imbalance interventions "
        "(hatched bars = SMOTE / SMOTE+Tomek)",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
