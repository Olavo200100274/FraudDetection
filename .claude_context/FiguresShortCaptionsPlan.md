# Plan: Short caption forms for the List of Figures (Maryam round-2, item 1/3)

**Created**: 2026-05-19
**Status**: Approved by user; execution in progress
**Origin**: Maryam Abbasi advisory note — Methodology, Results, and
Appendices currently have long figure captions appearing in full inside the
List of Figures, which makes the LoF cluttered and hard to skim. The
captions themselves are technically correct and Maryam did not ask for them
to be removed; she only asked for (a) confirmation that each long caption
is still appropriate and (b) addition of a short caption form so the LoF
becomes concise.

This is item 1 of 3 in Maryam's round-2 thesis polish; the other two items
(List of Tables, page references) are separate follow-ups.

## Context

Two things were requested per figure:

1. **Confirm the current long caption is correct.** A full Phase 1 audit
   was run across `Overleaf/Chapters/*.tex`. All 18 figures and 6
   subfigures were inventoried, with the current caption text, the
   surrounding context, and the cross-references verified. **Verdict for
   every long caption: KEEP_AS_IS** — no rewrites needed, no chapter prose
   depends on caption wording. Original long captions preserved bit-for-bit.
2. **Add a short caption form** via the LaTeX `[short]{long}` optional
   argument. The thesis uses `\captionof{figure}{...}` (in minipages,
   post-Step 13.5) and `\caption{...}` (in Appendix B's real
   `\begin{figure}[H]` floats); both accept the same optional-argument
   syntax. Currently **no figure has a short form**, including Fig 4.1
   (which the user described as an aspirational template, not the existing
   state).

User decision captured: subfigures in Appendix B also receive a `[short]`
argument so the LoF sub-entries are compact.

No experiments, results, numerical values, citations, BibTeX entries,
figures, tables, or scientific claims change. This is a presentation-only
edit limited to `\captionof{figure}` / `\caption` / `\subcaption` calls.

## Files affected

- `Overleaf/Chapters/4-Methodology.tex` — 3 captions
- `Overleaf/Chapters/5-Results.tex` — 13 captions
- `Overleaf/Chapters/7-appendices.tex` — 2 outer + 6 subcaption edits
- `.claude_context/PLAN.md` (new sub-step under PART 1, sibling to Step 8.1)
- `.claude_context/THESIS_STATE.md` (chapter status table notes)

No other files touched. No changes to BibTeX, glossary, `main.tex`, or any
content files outside captions.

## Proposed short forms

### Chapter 4 — Methodology (3 figures)

| Label | Proposed `[short]` |
|---|---|
| `fig:pipeline` | `End-to-end experimental pipeline for fraud detection` |
| `fig:cv_protocol` | `Leakage-free inner cross-validation protocol` |
| `fig:cross_domain` | `Cross-domain generalisation protocol within the BAF suite` |

### Chapter 5 — Results (13 figures)

| Label | Proposed `[short]` |
|---|---|
| `fig:prauc_ulb` | `Baseline PR-AUC by model on ULB 2013` |
| `fig:prauc_baf` | `Baseline PR-AUC by model on BAF Base` |
| `fig:heatmap_ulb_prauc` | `$\Delta$PR-AUC heatmap on ULB 2013` |
| `fig:heatmap_ulb_f2` | `$\Delta F_2$ heatmap on ULB 2013` |
| `fig:heatmap_baf_prauc` | `$\Delta$PR-AUC heatmap on BAF Base` |
| `fig:heatmap_baf_f2` | `$\Delta F_2$ heatmap on BAF Base` |
| `fig:shap_global` | `Top-15 SHAP features by model on BAF Base` |
| `fig:shap_beeswarm` | `SHAP beeswarm plot for LGBM on BAF Base` |
| `fig:shap_local` | `Local SHAP explanations for TP, FP, and FN cases (LGBM)` |
| `fig:shap_dependence` | `SHAP dependence plots for the top-3 LGBM features` |
| `fig:attention_aggregate` | `Mean per-token attention on BAF Base (FT-Transformer)` |
| `fig:attention_heatmap_fp` | `Attention matrix for a borderline False Positive case` |
| `fig:pr_curves_consolidated` | `Precision--Recall curves for all models on ULB and BAF` |

### Chapter 7 — Appendix B (2 outer + 6 subcaptions)

| Label / location | Type | Proposed `[short]` |
|---|---|---|
| `fig:attention_bar_fp` | outer caption | `Per-feature attention for three False Positive cases` |
| subfigure (a) of `fig:attention_bar_fp` | subcaption | `FP Case 1` |
| subfigure (b) of `fig:attention_bar_fp` | subcaption | `FP Case 2` |
| subfigure (c) of `fig:attention_bar_fp` | subcaption | `FP Case 3` |
| `fig:attention_bar_fn` | outer caption | `Per-feature attention for three False Negative cases` |
| subfigure (a) of `fig:attention_bar_fn` | subcaption | `FN Case 1` |
| subfigure (b) of `fig:attention_bar_fn` | subcaption | `FN Case 2` |
| subfigure (c) of `fig:attention_bar_fn` | subcaption | `FN Case 3` |

## Style principles applied

- **6–10 words per short form** (with one ~10-word outlier for the
  consolidated PR-curves figure).
- **Series parallelism**: the four heatmaps in §5.3–§5.6 share the template
  `<metric> heatmap on <dataset>`; the two baseline bar charts in §5.1–§5.2
  share `Baseline PR-AUC by model on <dataset>`; the FP/FN appendix
  subfigures use `<type> Case <n>`.
- **No restatement** of details already in the long caption (colour
  conventions, exact thresholds, sample counts, error-bar definitions,
  diagnostic-pattern names). The long caption is the data; the short form
  is the LoF headline.

## Execution order

1. Apply the 3 edits in `4-Methodology.tex`.
2. Apply the 13 edits in `5-Results.tex`.
3. Apply the 2 outer + 6 subcaption edits in `7-appendices.tex`.
4. Update `.claude_context/PLAN.md` (new sub-step under PART 1) and
   `.claude_context/THESIS_STATE.md` chapter status table notes.
5. Commit + push (single commit; cohesive presentation-only pass).

## Verification

- **LaTeX compile** in Overleaf: zero new warnings, no unresolved
  cross-references.
- **List of Figures inspection**: each entry shows the proposed short form
  (concise), not the long caption.
- **In-page captions** unchanged.
- **Cross-reference walk**: `grep -n '\\ref{fig:'` returns the same set
  before and after.

## Out of scope

- **Tables** — Maryam's second item; separate follow-up.
- **Page references** — Maryam's third item; separate follow-up.
- **Long caption text** — preserved bit-for-bit.
- **Article 1 and Article 2** — unaffected.
- **References, glossary, BibTeX, main.tex** — untouched.
