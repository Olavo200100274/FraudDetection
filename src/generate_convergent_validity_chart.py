"""
Generate Article-2 convergent-validity chart.

Two-panel horizontal bar chart:
  Left  --- LGBM top-10 features by mean |SHAP| on BAF Base.
  Right --- FT-Transformer top-10 tokens by mean attention on BAF Base.

Features that appear in BOTH top-10 lists are shaded identically across the
two panels (highlight colour), making the SHAP <-> Attention overlap visually
obvious. The convergent validity claim of the paper rests on this overlap.

Data sources:
  - results/baf_base/lgbm/none/run_20260312_085758/shap_global.json
  - results/baf_base/fttransformer/attention_analysis/attention_summary.json

Usage:
    cd src/
    python generate_convergent_validity_chart.py

Writes to: ../Article 2/figs/convergent_validity.pdf
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHAP_PATH = ROOT / "results" / "baf_base" / "lgbm" / "none" / "run_20260312_085758" / "shap_global.json"
ATTN_PATH = ROOT / "results" / "baf_base" / "fttransformer" / "attention_analysis" / "attention_summary.json"
OUT_PATH = ROOT / "Article 2" / "figs" / "convergent_validity.pdf"

HIGHLIGHT = "#2ca02c"   # green: features in BOTH top-10
SHAP_COLOR = "#1f77b4"  # LGBM blue (non-overlapping SHAP features)
ATTN_COLOR = "#ff7f0e"  # FT-T orange (non-overlapping attention features)


def load_top_n(d, n=10, exclude=("[CLS]",)):
    items = [(k, v) for k, v in d.items() if k not in exclude]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[:n]


def pretty(name):
    """Light cosmetic cleanup for axis labels."""
    return name.replace("_", " ")


def main():
    with open(SHAP_PATH, "r") as f:
        shap = json.load(f)
    with open(ATTN_PATH, "r") as f:
        attn_full = json.load(f)
    attention = attn_full["per_feature_mean_attention"]

    shap_top = load_top_n(shap, n=10)
    attn_top = load_top_n(attention, n=10)

    shap_features = [k for k, _ in shap_top]
    attn_features = [k for k, _ in attn_top]
    overlap = set(shap_features) & set(attn_features)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 5.6))

    # ---- Left panel: SHAP (LGBM) ----
    names_l = [pretty(k) for k, _ in shap_top]
    vals_l = [v for _, v in shap_top]
    colors_l = [HIGHLIGHT if k in overlap else SHAP_COLOR for k, _ in shap_top]
    y_l = np.arange(len(names_l))[::-1]
    ax_l.barh(y_l, vals_l, color=colors_l, edgecolor="black", linewidth=0.7)
    ax_l.set_yticks(y_l)
    ax_l.set_yticklabels(names_l, fontsize=10)
    ax_l.set_xlabel("Mean $|$SHAP$|$", fontsize=11)
    ax_l.set_title("LGBM --- SHAP top-10", fontsize=12, fontweight="bold")
    ax_l.grid(True, axis="x", alpha=0.3)
    ax_l.set_axisbelow(True)

    # ---- Right panel: Attention (FT-T) ----
    names_r = [pretty(k) for k, _ in attn_top]
    vals_r = [v for _, v in attn_top]
    colors_r = [HIGHLIGHT if k in overlap else ATTN_COLOR for k, _ in attn_top]
    y_r = np.arange(len(names_r))[::-1]
    ax_r.barh(y_r, vals_r, color=colors_r, edgecolor="black", linewidth=0.7)
    ax_r.set_yticks(y_r)
    ax_r.set_yticklabels(names_r, fontsize=10)
    ax_r.set_xlabel("Mean attention weight", fontsize=11)
    ax_r.set_title("FT-Transformer --- Attention top-10", fontsize=12, fontweight="bold")
    ax_r.grid(True, axis="x", alpha=0.3)
    ax_r.set_axisbelow(True)

    # Custom legend on the figure (overlapping = green; method-specific = its own colour)
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=HIGHLIGHT, edgecolor="black",
              label=f"In both top-10 (n\\,=\\,{len(overlap)})"),
        Patch(facecolor=SHAP_COLOR, edgecolor="black", label="SHAP-only"),
        Patch(facecolor=ATTN_COLOR, edgecolor="black", label="Attention-only"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.04), frameon=True, fontsize=10)

    fig.suptitle(
        "Convergent validity: SHAP attribution (LGBM) vs.\\ attention weights (FT-Transformer) on BAF Base",
        fontsize=12, y=1.10,
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT_PATH}")
    print(f"     Overlap (top-10 in both methods): {sorted(overlap)}")


if __name__ == "__main__":
    main()
