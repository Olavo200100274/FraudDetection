"""
Generate Article-2 cross-domain PR-AUC line chart.

Shows PR-AUC for each model trained on BAF Base and evaluated on
Base (in-domain) + Variants I-V (zero-shot cross-domain transfer).
The key story: RF *improves* on 4 of 5 variants while all other models degrade.

Values are taken directly from the thesis cross-domain results tables.

Usage:
    cd src/
    python generate_crossdomain_chart.py

Writes to: ../Article 2/figs/crossdomain_prauc.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "Article 2" / "figs" / "crossdomain_prauc.pdf"

# PR-AUC from thesis Table (crossdomain_prauc)
# Columns: Base, Var I, Var II, Var III, Var IV, Var V
CROSSDOMAIN_PRAUC = {
    "LR":       [0.1432, 0.1084, 0.1420, 0.1015, 0.1359, 0.1005],
    "RF":       [0.1588, 0.2058, 0.2245, 0.1699, 0.2108, 0.1593],
    "LGBM":     [0.1768, 0.1546, 0.1856, 0.1338, 0.1718, 0.1222],
    "CatBoost": [0.1796, 0.1518, 0.1871, 0.1336, 0.1736, 0.1235],
    "FT-Trans.": [0.1797, 0.1476, 0.1770, 0.1296, 0.1656, 0.1179],
}

MODEL_COLORS = {
    "LR":       "#9467bd",
    "RF":       "#2ca02c",
    "LGBM":     "#1f77b4",
    "CatBoost": "#d62728",
    "FT-Trans.": "#ff7f0e",
}

MODEL_STYLES = {
    "LR":       {"linestyle": "--", "linewidth": 1.5, "marker": "o", "alpha": 0.7},
    "RF":       {"linestyle": "-",  "linewidth": 2.5, "marker": "s", "alpha": 1.0},
    "LGBM":     {"linestyle": "-",  "linewidth": 2.0, "marker": "^", "alpha": 0.9},
    "CatBoost": {"linestyle": "-",  "linewidth": 2.0, "marker": "D", "alpha": 0.9},
    "FT-Trans.": {"linestyle": "-.", "linewidth": 2.0, "marker": "v", "alpha": 0.9},
}

X_LABELS = ["Base\n(in-domain)", "Variant I", "Variant II", "Variant III", "Variant IV", "Variant V"]
X = np.arange(len(X_LABELS))


def main():
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    for model, values in CROSSDOMAIN_PRAUC.items():
        style = MODEL_STYLES[model]
        ax.plot(X, values, color=MODEL_COLORS[model], label=model,
                markersize=8, markeredgewidth=0.8, markeredgecolor="black",
                **style)

    # Annotate the RF story: add upward arrow annotation on Var II
    ax.annotate("RF improves\nacross variants",
                xy=(2, CROSSDOMAIN_PRAUC["RF"][2]),
                xytext=(2.4, 0.212),
                fontsize=8.5,
                color=MODEL_COLORS["RF"],
                arrowprops=dict(arrowstyle="->", color=MODEL_COLORS["RF"], lw=1.2),
                ha="left")

    ax.set_xticks(X)
    ax.set_xticklabels(X_LABELS, fontsize=10)
    ax.set_ylabel("PR-AUC", fontsize=12)
    ax.set_ylim([0.08, 0.26])
    ax.grid(True, alpha=0.3, axis="y")
    ax.axvline(x=0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.text(0.52, 0.245, "← in-domain | cross-domain →", fontsize=8, color="gray", va="top")

    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
