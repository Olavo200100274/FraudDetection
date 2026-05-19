# Plan: Short caption forms for the List of Tables (Maryam round 2, item 2/3)

**Created**: 2026-05-19
**Status**: Approved by user; execution in progress
**Origin**: Maryam Abbasi advisory note — the List of Tables was as
cluttered as the List of Figures had been before Step 8.2 because every
entry showed the full long caption. This is the direct counterpart of
Step 8.2 (item 1/3 — figures) applied to the thesis's tables; item 3/3
(page references) follows separately.

This is item 2 of 3 in Maryam's round-2 thesis polish.

## Context

Two things per table (mirrors the figures pass):

1. **Confirm the current long caption is correct.** A Phase 1 audit (1
   Explore agent) inventoried all 32 tables in
   `Overleaf/Chapters/*.tex`. Verdict: 31 KEEP_AS_IS and 1 MINOR_EDIT
   (Table 26 — decision framework — contains a meta-sentence with a
   `\ref{}` backreference that describes where the recommendations are
   derived rather than what the table shows; dropping it tightens the
   caption).
2. **Add a `[short]{long}` argument** so the List of Tables entry
   becomes concise. Style: 4–10 words per short form; parallel templates
   for the seven table series.

No experiments, results, numerical values, citations, BibTeX entries,
figures, tables, or scientific claims change. This is a presentation-only
edit limited to `\captionof{table}` calls (and one short long-caption
edit on Table 26).

## Files affected

- `Overleaf/Chapters/5-Results.tex` — 30 tables (29 short-form additions
  + Table 26 receives short form AND a small long-caption edit)
- `Overleaf/Chapters/7-appendices.tex` — 2 short-form additions
  (Appendix A hyperparameter tables)
- `.claude_context/PLAN.md` (new sub-step under PART 1, sibling to
  Step 8.1 and Step 8.2)
- `.claude_context/THESIS_STATE.md` (chapter status table notes)

No other files touched.

## Proposed short forms

### Chapter 5 — §5.1 Baseline (4 tables)

| Label | Proposed `[short]` |
|---|---|
| `tab:baseline_ulb` | `Baseline metrics on ULB 2013` |
| `tab:ops_ulb` | `Baseline operational metrics on ULB 2013` |
| `tab:baseline_baf` | `Baseline metrics on BAF Base` |
| `tab:ops_baf` | `Baseline operational metrics on BAF Base` |

### Chapter 5 — §5.2 Threshold Sensitivity (8 tables)

Template: `<Metric> by threshold rule on <Dataset>`.

| Label | Proposed `[short]` |
|---|---|
| `tab:threshold_ulb_f2` | `$F_2$ by threshold rule on ULB 2013` |
| `tab:threshold_ulb_f1` | `$F_1$ by threshold rule on ULB 2013` |
| `tab:threshold_ulb_recall` | `Recall by threshold rule on ULB 2013` |
| `tab:threshold_ulb_alert` | `Alert Rate by threshold rule on ULB 2013` |
| `tab:threshold_baf_f2` | `$F_2$ by threshold rule on BAF Base` |
| `tab:threshold_baf_f1` | `$F_1$ by threshold rule on BAF Base` |
| `tab:threshold_baf_recall` | `Recall by threshold rule on BAF Base` |
| `tab:threshold_baf_alert` | `Alert Rate by threshold rule on BAF Base` |

### Chapter 5 — §5.3 Full Factorial (6 tables)

Template: `<Metric> factorial on <Dataset>`.

| Label | Proposed `[short]` |
|---|---|
| `tab:factorial_ulb_prauc` | `PR-AUC factorial on ULB 2013` |
| `tab:factorial_ulb_rocauc` | `ROC-AUC factorial on ULB 2013` |
| `tab:factorial_ulb_f2` | `$F_2$ factorial on ULB 2013` |
| `tab:factorial_baf_prauc` | `PR-AUC factorial on BAF Base` |
| `tab:factorial_baf_rocauc` | `ROC-AUC factorial on BAF Base` |
| `tab:factorial_baf_f2` | `$F_2$ factorial on BAF Base` |

### Chapter 5 — §5.4 Transformer Robustness (2 tables)

| Label | Proposed `[short]` |
|---|---|
| `tab:transformer_robustness_ulb` | `FT-Transformer vs.\ CatBoost robustness on ULB 2013` |
| `tab:transformer_robustness_baf` | `FT-Transformer vs.\ CatBoost robustness on BAF Base` |

### Chapter 5 — §5.5 Cross-Domain Generalisation (3 tables)

Template: `Cross-domain <Metric> on BAF Variants`.

| Label | Proposed `[short]` |
|---|---|
| `tab:crossdomain_prauc` | `Cross-domain PR-AUC on BAF Variants` |
| `tab:crossdomain_f2` | `Cross-domain $F_2$ on BAF Variants` |
| `tab:crossdomain_rocauc` | `Cross-domain ROC-AUC on BAF Variants` |

### Chapter 5 — §5.6 Interpretability (2 tables)

| Label | Proposed `[short]` |
|---|---|
| `tab:shap_consistency` | `Cross-model SHAP top-10 Jaccard agreement` |
| `tab:shap_stability` | `Cross-variant SHAP top-10 stability for LGBM` |

### Chapter 5 — §5.7 Consolidated Summary (5 tables)

| Label | Proposed `[short]` |
|---|---|
| `tab:decision_framework` | `Decision framework by deployment scenario` |
| `tab:ci_ulb` | `Bootstrap PR-AUC and ROC-AUC CIs on ULB 2013` |
| `tab:ci_baf` | `Bootstrap PR-AUC and ROC-AUC CIs on BAF Base` |
| `tab:cost_ulb` | `Computational cost on ULB 2013` |
| `tab:cost_baf` | `Computational cost on BAF Base` |

### Chapter 7 — Appendix A Hyperparameters (2 tables)

| Label | Proposed `[short]` |
|---|---|
| `tab:hyperparams_ulb` | `Best Optuna hyperparameters on ULB 2013` |
| `tab:hyperparams_baf` | `Best Optuna hyperparameters on BAF Base` |

## Minor edit to Table 26 long caption

Current: *"Decision framework: recommended configuration by deployment
scenario. Recommendations are derived from the experimental results
presented in Sections~\ref{sec:baseline_results}--\ref{sec:shap_results}."*

Replaced with: *"Decision framework: recommended (model, imbalance
strategy, threshold rule) configuration by deployment scenario."*

Rationale: the trailing meta-sentence describes the recommendations'
provenance rather than the table's content; the same information appears
in the prose immediately above the table. The added `(model, imbalance
strategy, threshold rule)` parenthetical mirrors the three result
columns so the caption tells the reader what the rows decide on.

## Style principles applied

- **4–10 words per short form**.
- **Series parallelism** across the seven series so the LoT reads as a
  coherent index.
- **No restatement** of meta-information already in the long caption
  (e.g. `(strategy = None)`, `(50 trials, 5-fold CV)`, `(1\,000
  iterations)` are dropped from the short form).

## Execution order

1. Apply 30 caption edits in `5-Results.tex` (including the long-caption
   edit on Table 26).
2. Apply 2 caption edits in `7-appendices.tex` (Appendix A
   hyperparameter tables).
3. Update `.claude_context/PLAN.md` and `.claude_context/THESIS_STATE.md`.
4. Single commit.

## Verification

- **LaTeX compile** in Overleaf: zero new warnings, no unresolved
  cross-references.
- **List of Tables inspection** in the generated PDF: each entry shows
  the proposed short form (concise), one line per table.
- **In-page captions** unchanged (except for Table 26's long-caption
  edit).
- **Cross-reference walk**: `grep -n '\\ref{tab:'` returns the same set
  before and after.

## Out of scope

- **Item 3/3 of Maryam round 2** — page references where appropriate;
  separate follow-up.
- **Long caption text for tables 1–25 and 27–32** — preserved
  bit-for-bit.
- **Article 1 and Article 2** — unaffected.
- **References, glossary, BibTeX, main.tex** — untouched.
