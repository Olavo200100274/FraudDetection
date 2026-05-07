# THESIS_STATE.md
**Last updated**: 2026-05-06 (Conclusion written — all main chapters complete)

---

## Overleaf structure

```
Overleaf/
├── main.tex                      # document class, chapter inputs
├── glossary.tex                  # ✅ abbreviations glossary (37 entries — fraud detection specific)
├── references.bib                # 43 BibTeX entries (see REFERENCES_STATE.md)
├── include/
│   ├── preamble.tex              # all \usepackage declarations
│   ├── 0-titlepage-EN.tex        # ✅ COMPLETE (title, author, supervisors, Junho 2026)
│   ├── acknowledgments.tex       # ✅ COMPLETE (supervisors + colleagues + professors)
│   ├── epigraph.tex              # ✅ Donald Knuth quote (kept as-is)
│   └── copyright.tex             # ✅ standard PT+EN text (no changes needed)
├── Chapters/
│   ├── abstract/abstract-EN.tex  # ❌ PLACEHOLDER (needs writing + fix wrong keywords)
│   ├── abstract/abstract-PT.tex  # ❌ PLACEHOLDER (Lorem Ipsum)
│   ├── 1-Introduction.tex        # ✅ COMPLETE
│   ├── 2-State of the Art.tex    # ✅ COMPLETE
│   ├── 3-Methodology.tex         # ✅ COMPLETE
│   ├── 4-Results.tex             # ✅ COMPLETE
│   ├── 5-conclusion.tex          # ✅ COMPLETE
│   └── 6-appendices.tex          # ❌ PLACEHOLDER (Lorem Ipsum — decide or remove)
├── figures/baf/                  # 9 PDF figures + 3 PNG attention figures
├── figures/ulb/                  # 5 PDF figures
├── references/                   # curated PDF library (see REFERENCES_STATE.md)
└── tables/                       # (no longer used — tables inline in Results.tex)
```

---

## Chapter status

| Chapter | Status | Notes |
|---------|--------|-------|
| Abstract EN | ✅ | Complete — 4 paragraphs, ~280 words, 9 keywords |
| Abstract PT | ✅ | Complete — faithful translation of EN |
| 1-Introduction | ✅ | 5 sections: Context, Motivation, Objectives+RQs (×5), Research Approach, Document Structure. 8 citations. |
| 2-State of the Art | ✅ | 7 sections, 43 BibTeX entries. Sec 5.3 carries Maryam's GBDT-vs-Transformer framing. Sec 7 maps 6 gaps to 6 contributions. |
| 3-Methodology | ✅ | 11 sections, 3 TikZ figures (pipeline, leakage-free CV, cross-domain). Model and strategy selection now justified. |
| 4-Results | ✅ | 9 sections: baseline, threshold, factorial, transformer robustness, cross-domain, SHAP, **attention diagnostics (Sec 7 — NEW)**, consolidated, discussion. 4 factual errors corrected. |
| 5-Conclusion | ✅ | 4 sections: contributions summary, RQ answers (×5 with numbers), limitations (×5), future work (×5 concrete directions). |
| 6-Appendices | ✅ | Complete — Appendix A: hyperparameter tables (ULB+BAF, all models); Appendix B: 6 attention bar charts (FP/FN cases) |

---

## The 6 thesis contributions (to defend)

1. **Unified leakage-free comparison** — classical ML vs FT-Transformer across ULB + BAF, single reproducible protocol
2. **Threshold selection as first-class experimental variable** — 4 strategies, selected on validation never on test
3. **Full factorial 7×5×2** — imbalance strategy × model × dataset, structured analysis of interactions
4. **Cross-domain generalization** — BAF Base→Variants I-V without retraining, distribution shift study
5. **SHAP cross-model + cross-variant consistency** — Jaccard analysis (LGBM/CatBoost=1.00, stable across all 6 BAF variants)
6. **FT-Transformer attention diagnostics** — Foggy Vision (entropy=0.976), confirms 0.18 PR-AUC is dataset ceiling not pathology

---

## Key experimental findings (for Abstract writing)

- ULB: CatBoost 0.885 > LGBM 0.874 > RF 0.867 > FT-T 0.856 > LR 0.742 >> OCSVM 0.335
- BAF: FT-T ties CatBoost (both 0.180) — gap closes at scale
- Threshold: fixed τ=0.5 catastrophic on BAF (F2≈0); max-F2 recovers ~9× improvement
- SMOTE: degrades FT-T 41% on BAF vs 14% CatBoost — architecture-specific interaction
- Class Weights: safest strategy across all models
- RF: most cross-domain robust (improves on 4/5 BAF variants despite lowest in-domain score)
- SHAP Jaccard LGBM/CatBoost=1.00, stable across all 6 variants; FT-T=0.54 vs trees
- Foggy Vision: normalised entropy=0.976, no pathological attention pattern

---

## Advisor directives (Maryam Abbasi)

### Central narrative (confirmed Discord 2026-05-04)
> "The article is not about a better transformer. The idea is to show **why a simple LightGBM outperforms the transformer**. Consider this when completing the state of the art."

### Key implications for writing
- Abstract: frame around GBDT vs Transformer comparison, NOT around "we built a system"
- The 3 GBDT advantages to highlight: (a) calibration, (b) robustness to imbalance strategies (SMOTE 41% vs 14%), (c) no architectural mismatch (Transformer built for sequences, tabular ≠ sequential)
- Foggy Vision is the mechanistic explanation: FT-T finds all available signal but signal is limited → dataset ceiling

### Two papers (revised 2026-05-07 — Discord)
Originally a single TPAMI article. Maryam revised the plan: split into **two distinct papers** by topic and audience.

| | **Paper 1 — Benchmark + Threshold** | **Paper 2 — Mechanistic / Robustness** |
|---|---|---|
| Folder | `Article 1/` | `Article 2/` |
| Centrepiece | Threshold sensitivity (9× F₂ swing on BAF) | FT-Transformer attention diagnostics (Foggy Vision) |
| Owns Results §§ | 1, 2, 3 + bootstrap CIs + cost tables | 4, 5, 6, 7 + Discussion |
| Audience | Practitioners | ML/AI research community |
| Target venues | Expert Systems with Applications / DSS / Information Sciences (all Elsevier CAS) | IEEE TNNLS / Neural Networks / Applied Soft Computing |
| Length | 10–12K words, table-heavy | 9–11K words, figure-heavy |
| Working title | "Threshold Selection as a Critical Design Choice in Financial Fraud Detection..." | "Why LightGBM Generalises Better Than Tabular Transformers Under Distribution Shift..." |
| Plan part | Part 2 (Steps 9–14) | Part 3 (Steps 15–20) |

**Writing order:** Paper 1 first (establishes leakage-free protocol + decision framework), then Paper 2 (cites Paper 1 for shared protocol).

---

## SoA structure (FROZEN — written 2026-05-05)

1. Fraud Detection in Financial Transactions (context + datasets)
2. Evaluation of Fraud Detection Systems under Class Imbalance (PR-AUC, F-β, threshold, calibration)
3. Classical Machine Learning for Fraud Detection (LR, RF, GBDT families)
4. Deep Learning for Fraud Detection (MLP, RNN, autoencoders — brief)
5. Transformers for Tabular Fraud Detection ⭐ (TabTransformer, SAINT, FT-T, DL-vs-GBDT debate)
6. Explainable AI in Fraud Detection (SHAP, LIME, cross-model consistency)
7. Synthesis, Limitations of Existing Work, and Research Gaps (6 gaps → 6 contributions)

---

## LaTeX known issues (all resolved)

- TikZ `step` reserved key → renamed to `procstep` in Methodology ✅
- Duplicate `\usepackage` declarations → consolidated in preamble.tex ✅
- `\blankpage` undefined → resolved ✅
- Chapter hierarchy → limited to 3 levels (x.x.x) via secnumdepth=3 ✅
- `\paragraph` used only for inline bold headings (not in TOC) ✅
- Missing BibTeX citations (Optuna, FT-Transformer, SHAP, LightGBM, CatBoost) → all added ✅

---

## Git repository

- Remote: `https://github.com/Olavo200100274/FraudDetection.git`
- Branch: `main`
- Last commit: `624ff0a` (Step 5: Write Conclusion chapter)
- Git LFS active for: `*.joblib`, `*.pt`, `*.npy`, `datasets/*.csv`
