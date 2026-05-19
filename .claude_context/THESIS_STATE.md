# THESIS_STATE.md
**Last updated**: 2026-05-19 (after Article 1 Maryam round-1 writing-only revision pass)

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
| 4-Results | ✅ | 9 sections: baseline, threshold, factorial, transformer robustness, cross-domain, SHAP, attention diagnostics, consolidated, **synthesis + threats to validity (renamed from "discussion" in Step 13.5)**. Prose revised so discussion is integrated inline rather than collected into per-section `\paragraph{Discussion.}` blocks. 30 tables, 13 figures, 43 labels. |
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
- **Float drift (2026-05-11)** — tables and figures were appearing far from their definition. Fix: convert every `\begin{table}/\begin{figure}` to `\noindent\begin{minipage}{\linewidth}\centering ... \end{minipage}` + `\captionof{table/figure}{...}`. Minipages are not floats so they appear exactly where placed. Exception: Appendix B's two subfigure groups stay as `\begin{figure}[H]` (with `float` package, already in preamble), because `subcaption` errors out when `subfigure` is used outside a real float ✅
- **Caption hypcap warnings (2026-05-11)** — `\captionof` inside a minipage triggers "Package caption Warning: hypcap=true will be ignored" for every caption. Fix: add `\captionsetup{hypcap=false}` to `preamble.tex` after the `caption` package is loaded ✅

---

## Articles status

### Article 1 — Practitioner / Threshold-centric (Paper 1)
- **Folder**: `Article 1/`
- **Title (frozen)**: *"Threshold Selection as a Critical Design Choice in Financial Fraud Detection: A Systematic Comparison of Classical Machine Learning Models and Tabular Transformers"*
- **Short title**: *"Threshold Selection in Financial Fraud Detection"*
- **Venue**: Expert Systems with Applications (Elsevier, Scopus Q1)
- **Template**: Elsevier CAS single-column (`cas-sc.cls`)
- **Main file**: `Article 1/main.tex` (writing-only revised draft as of Step 13.6, 2026-05-19; pre-revision version preserved as `Article 1/main_old.tex`)
- **Bibliography**: `Article 1/cas-refs.bib` (47 entries — thesis 43 + 4 added in Step 13.5: tibshirani1996regression, breiman2001random, scholkopf2001estimating, fernandez2018smote)
- **Figures**: `Article 1/figs/` — includes `ulb_pr_curves_threshold.pdf`, `baf_pr_curves_threshold.pdf`, factorial heatmaps, decision-framework graphic
- **Owns thesis content**: Results §1 (baseline) + §2 (threshold sensitivity, **centrepiece**) + §3 (full factorial) + §8 (bootstrap CIs + cost tables + decision framework)
- **Status**: ✅ first draft → ✅ Maryam round-1 feedback ("too much itemize") absorbed via writing-only revision pass (Step 13.6); back in Maryam review queue (Step 14 still pending — second review of revised draft)
- **Reusable revision prompt**: stored verbatim in `.claude_context/RevisingPaperContext.md` (Maryam-authored). Applies to Article 2 with two section-specific adaptations (see header of that file).

### Article 2 — Mechanistic / "Simple Trees Suffice" (Paper 2)
- **Folder**: `Article 2/`
- **Title (frozen 2026-05-14 after Step 19.5 reframe)**: *"Why Simple Tree-Based Models Suffice for Financial Fraud Detection: Mechanistic Evidence from SHAP–Attention Convergence and Cross-Domain Transfer"*
- **Short title**: *"Why Simple Trees Suffice for Financial Fraud Detection"*
- **Venue**: Applied Soft Computing (Elsevier, Scopus Q1) — switched from Neural Networks during reframe
- **Template**: Elsevier CAS single-column (`cas-sc.cls`)
- **Main file**: `Article 2/main.tex` (~9K words, 8 main figures, 7 main tables)
- **Bibliography**: `Article 2/cas-refs.bib` (49 entries — Article 1's 47 + grinsztajn2022tree + caixeiro2025threshold placeholder)
- **Figures**: `Article 2/figs/`
  - `crossdomain_prauc.pdf` — RF cross-domain advantage line chart
  - `smote_asymmetry.pdf` — FT-T vs CatBoost across 7 strategies × 2 datasets
  - `shap_global.pdf`, `shap_beeswarm.pdf`, `shap_waterfall.pdf` — SHAP attribution
  - `convergent_validity.pdf` — LGBM SHAP top-10 vs FT-T attention top-10 with overlap shading
  - `attention_aggregate.png`, `attention_heatmap_fp.png` — Foggy Vision diagnostic
- **Owns thesis content**: Results §4 (transformer robustness) + §5 (cross-domain) + §6 (SHAP) + §7 (attention diagnostics)
- **Central argument**: 5-pillar "Simple Trees Suffice" — (1) no resampling penalty, (2) cross-domain parity/advantage, (3) stable explanations, (4) no headroom above ceiling, (5) ~20× training / ~37× tuning cheaper
- **Methodological novelty**: convergent validity (SHAP↔Attention) imported from psychometric measurement theory — 5/10 features overlap, top-2 identical (`device_os`, `housing_status`)
- **Status**: ✅ first draft (Step 19) → reframe (Step 19.5) → deep-review pass (commit `6c25574`) complete; awaiting Maryam review (Step 20 pending)
- **Rejected interim titles**:
  - "Mechanistic Understanding of Tabular Transformer Limitations..." (too transformer-centric; rejected 2026-05-14 morning)
  - "Why LightGBM Generalises Better Than Tabular Transformers..." (the plan's original direction; rejected for being factually weak — LGBM does not literally beat FT-T cross-domain; RF is the real winner; CatBoost is the SMOTE-robust hero)

---

## Git repository

- Remote: `https://github.com/Olavo200100274/FraudDetection.git`
- Branch: `main`
- Last commit: `6c25574` (Article 2 deep-review pass: fix 7 factual inconsistencies)
- Recent history (newest → oldest):
  - `6c25574` Article 2 deep-review pass: fix 7 factual inconsistencies
  - `1782c46` Step 19.5: Reframe Article 2 to "Simple Trees Suffice" + venue switch
  - `9b2d912` Steps 15–19: Article 2 full first draft (mechanistic paper)
  - `c5646e8` Article 1: pre-submission tweaks (decision framework framing + Future Work)
  - `bd02145` Article 1: align section 4 prose with new PR-curves figure
- Git LFS active for: `*.joblib`, `*.pt`, `*.npy`, `datasets/*.csv`
