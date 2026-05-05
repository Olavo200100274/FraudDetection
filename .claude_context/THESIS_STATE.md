# THESIS_STATE.md
**Last updated**: 2026-05-05

---

## Overleaf structure

```
Overleaf/
├── main.tex                      # document class, chapter inputs
├── references.bib                # 41 BibTeX entries (see REFERENCES_STATE.md)
├── include/preamble.tex          # all \usepackage declarations
├── Chapters/
│   ├── abstract/abstract-EN.tex  # ❌ PLACEHOLDER (needs writing + fix wrong keywords)
│   ├── abstract/abstract-PT.tex  # ❌ PLACEHOLDER (Lorem Ipsum)
│   ├── 1-Introduction.tex        # ❌ EMPTY (only section headings)
│   ├── 2-State of the Art.tex    # ❌ EMPTY (only section headings)
│   ├── 3-Methodology.tex         # ✅ COMPLETE
│   ├── 4-Results.tex             # ✅ COMPLETE
│   ├── 5-conclusion.tex          # ❌ INCOMPLETE (1 sentence only)
│   └── 6-appendices.tex          # ❌ PLACEHOLDER (Lorem Ipsum)
├── figures/baf/ + figures/ulb/   # pre-generated PDF figures
├── references/                   # curated PDF library (see REFERENCES_STATE.md)
└── tables/                       # (no longer used — tables inline in Results.tex)
```

---

## Chapter status

| Chapter | Status | Notes |
|---------|--------|-------|
| Abstract EN | ❌ | Wrong keywords (drug discovery remnant), needs full rewrite |
| Abstract PT | ❌ | Lorem Ipsum placeholder |
| 1-Introduction | ❌ | Empty — to write AFTER SoA (depends on it for motivation) |
| 2-State of the Art | ❌ | Empty — NEXT MAJOR TASK |
| 3-Methodology | ✅ | Complete, 3 TikZ figures (pipeline, leakage-free CV, cross-domain) |
| 4-Results | ✅ | Complete — baseline, threshold, factorial, FT-T robustness, cross-domain, SHAP, consolidated, discussion, threats to validity |
| 5-Conclusion | ❌ | 1 sentence — needs: contributions summary, RQ answers, limitations, future work |
| 6-Appendices | ❌ | Lorem Ipsum — decide: keep with real content or remove |

---

## The 6 thesis contributions (to defend)

1. **Unified leakage-free comparison** classical ML vs FT-Transformer across 2 datasets with single reproducible protocol
2. **Threshold selection as first-class experimental variable** — 4 strategies, selected on validation never on test
3. **Full factorial 7×5×2** imbalance strategy × model × dataset — structured analysis of interactions
4. **Cross-domain generalization** BAF Base→Variants I-V without retraining — distribution shift study
5. **SHAP cross-model + cross-variant consistency** — Jaccard analysis proving feature importance stability
6. **FT-Transformer attention diagnostics** — entropy/pattern analysis confirming Foggy Vision, not pathological

---

## Advisor directives (Maryam Abbasi)

### Central narrative (confirmed Discord 2026-05-04)
> "The article is not about a better transformer. The idea is to show **why a simple LightGBM outperforms the transformer**. Consider this when completing the state of the art."

### Key implications for writing
- SoA must build up to question: "Why does GBDT match/beat sophisticated tabular transformers on fraud?"
- Discussion must argue with evidence: GBDT wins because of (a) calibration, (b) robustness to imbalance strategies, (c) no architectural mismatch (Transformer built for sequences, tabular ≠ sequential)
- Abstract: frame around GBDT vs Transformer comparison, NOT around "we built a system"

### Article (IEEE TPAMI target)
- Maryam said: "lets try to publish it here: ieee transactions on pattern analysis and machine intelligence"
- TPAMI is top-tier Q1, ~20-25% acceptance, 6+ months review, ~14 pages
- Tone: measured, rigorous (NOT provocative like "DL Is Not All You Need" title)
- Title direction: "A Leakage-Free Comparative Study of Classical ML and Tabular Transformers for Financial Fraud Detection"
- Article comes AFTER thesis completion (Part 2 of plan)

---

## Proposed SoA structure (to freeze in Step 2 of plan)

1. **Fraud Detection in Financial Transactions** — context + datasets (Jesus 2022 BAF, Pozzolo 2015 ULB)
2. **Evaluation under Extreme Class Imbalance** — PR-AUC vs ROC-AUC, F-β metrics, threshold, calibration
3. **Classical Machine Learning for Fraud Detection** — LR, RF, GBDT families, strengths/limits
4. **Deep Learning for Tabular Fraud Detection** ⭐ — TabTransformer, SAINT, FT-Transformer, tabular DL debate
5. **Explainable AI in Fraud Detection** — SHAP, LIME, accuracy-vs-interpretability trade-off
6. **Synthesis: Limitations and Research Gaps** — maps 6 thesis contributions to gaps in literature

This structure NOT yet frozen — Step 2 of the plan decides it.

---

## LaTeX known issues (resolved)

- TikZ `step` reserved key → renamed to `procstep` in Methodology ✅
- Duplicate `\usepackage` declarations → consolidated in preamble.tex ✅
- `\blankpage` undefined → resolved ✅
- Chapter hierarchy → limited to 3 levels (x.x.x) via secnumdepth=3 ✅
- `\paragraph` used only for inline bold headings (not in TOC) ✅
- 4 missing BibTeX citations (Optuna, FT-Transformer, SHAP) → added ✅

---

## Git repository
- Remote: `https://github.com/Olavo200100274/FraudDetection.git`
- Branch: `main`
- Last commit: `9117b63` (Phase 2 Part 2: ULB references curation final 23 of 57 PDFs)
- Git LFS active for: `*.joblib`, `*.pt`, `*.npy`, `datasets/*.csv`
