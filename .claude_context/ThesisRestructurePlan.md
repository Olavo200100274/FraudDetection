# Thesis Restructure Plan — "Theoretical Background" chapter

**Created**: 2026-05-19
**Status**: Approved by user; execution in progress
**Origin**: Maryam Abbasi advisory note — Methodology chapter currently mixes
theory (what each tool is) with application (how we used it in our project).
She asked for a new chapter to hold the theory, so that the Methodology can be
trimmed to project-specific protocol only.

This file is the persistent record of the design decision. It mirrors the live
plan that was executed; the actual edits, commits, and any deviations made
during execution are tracked in `.claude_context/PLAN.md` (the sequential plan
log) and in the git history.

## Context

The current Methodology chapter (`Overleaf/Chapters/3-Methodology.tex`) is doing
two jobs at once: it explains *what each tool is* (textbook-style theory) and
*how the tool was applied to this project*. A Methodology chapter should do only
the second; the first belongs in a separate theoretical foundations chapter
that sits between the State of the Art and the Methodology.

This restructure splits the Methodology into two chapters:

- A new **"Theoretical Background"** chapter that holds the toolkit definitions
  (algorithms, metric formulas, attention mechanism, SHAP/Shapley axioms, TPE,
  etc.) once each. Distinct from the State of the Art: SoA covers the
  *literature debate* (e.g. the GBDT-vs-Transformer discussion, the
  threshold-neglect critique); the new chapter covers the *tool definitions*
  themselves.
- A trimmed Methodology that keeps everything project-specific (datasets,
  leakage-free protocol, exact HPO budget, cross-domain protocol, etc.) and
  cross-references the new chapter whenever a theoretical refresher is needed.

User decisions:
- **Chapter name**: *Theoretical Background*.
- **File numbering**: renumber all subsequent chapter files so filename index
  matches chapter order.

No experiments, results, numbers, tables, figures, citations, or scientific
claims change. This is a structural prose-and-organisation refactor.

## Files affected

### Created
- `Overleaf/Chapters/3-Theoretical Background.tex` — new chapter.

### Renamed (git-mv to keep history)
- `Overleaf/Chapters/3-Methodology.tex` → `Overleaf/Chapters/4-Methodology.tex`
- `Overleaf/Chapters/4-Results.tex` → `Overleaf/Chapters/5-Results.tex`
- `Overleaf/Chapters/5-conclusion.tex` → `Overleaf/Chapters/6-conclusion.tex`
- `Overleaf/Chapters/6-appendices.tex` → `Overleaf/Chapters/7-appendices.tex`

### Edited
- `Overleaf/main.tex` — update the `\input{}` block to reflect the new chapter
  and the renamed files (lines 65–70 region).
- `Overleaf/Chapters/4-Methodology.tex` (post-rename) — extract theoretical
  passages and replace each with a short pointer to the new chapter; keep all
  project-specific protocol untouched.
- `Overleaf/Chapters/5-Results.tex` (post-rename) — update the one cross-ref
  identified by the audit (`\ref{sec:metrics}` on line 26 may need to point to
  the new chapter's metrics section instead of Methodology).
- `Overleaf/include/abbreviation.tex` (glossary) — add missing acronym entries
  the new chapter will introduce.

### Context files (maintenance per project policy)
- `.claude_context/PLAN.md` — add as a new step.
- `.claude_context/THESIS_STATE.md` — update chapter status table and renumber.

## New chapter structure (`3-Theoretical Background.tex`)

Section order mirrors the order tools appear in the (post-rewrite) Methodology,
so cross-references resolve linearly.

1. **Chapter introduction** (½ page) — distinguish from SoA (toolkit
   definitions, not literature debate) and from Methodology (general theory of
   each tool, not our specific use).
2. **Supervised models** (`sec:tb:models`) — LR, RF, GBDT (LightGBM + CatBoost),
   OCSVM, FT-Transformer.
3. **Class-imbalance handling techniques** (`sec:tb:imbalance`) — SMOTE,
   SMOTE+Tomek, SMOTEENN, RUS, ROS, Class Weights.
4. **Evaluation metrics** (`sec:tb:metrics`) — Confusion-matrix primitives,
   Precision, Recall, F-beta, PR-AUC vs. ROC-AUC, calibration.
5. **Decision-threshold selection** (`sec:tb:thresholds`) — score-to-decision
   mapping; four classes of threshold rule as general objects.
6. **Hyperparameter optimisation** (`sec:tb:hpo`) — Bayesian optimisation and
   TPE algorithm; pruning at a conceptual level.
7. **SHAP and feature attribution** (`sec:tb:shap`) — Shapley values, SHAP
   axioms, TreeSHAP, LinearExplainer, GradientExplainer, Jaccard similarity as
   an attribution-comparison metric.
8. **Attention diagnostics** (`sec:tb:attention`) — *conditional*. Include only
   if attention-entropy theory currently lives outside the new chapter and can
   be cleanly migrated.

The chapter does NOT include datasets, splits, fold counts, Optuna trial
budgets, seeds, hardware, or any specific numerical configuration — those all
stay in the Methodology.

## Methodology rewrite (`4-Methodology.tex` after rename)

For each currently-MIXED subsection identified by the Phase 1 audit, extract
theory to the new chapter and replace with a short pointer.

| Methodology section | Action |
|---|---|
| `sec:objective` — Research Objective | Untouched. |
| `sec:datasets` — Datasets | Untouched. |
| `sec:models` — Models | Replace per-model textbook descriptions with selection rationale + one pointer to `\ref{sec:tb:models}`. |
| `sec:imbalance_strategies` — Imbalance handling | Replace algorithm-by-algorithm definitions with the factorial-design statement + pointer to `\ref{sec:tb:imbalance}`. |
| `sec:protocol` — Leakage-free protocol | Untouched. |
| `sec:threshold_study` — Threshold selection study | Replace conceptual $F_\beta$ motivation with "which four rules we evaluate and how each $\tau$ is selected"; pointer to `\ref{sec:tb:thresholds}` and `\ref{sec:tb:metrics}`. |
| `sec:tuning` — Hyperparameter tuning | Keep the specific tuning protocol; strip "what TPE is"; pointer to `\ref{sec:tb:hpo}`. |
| `sec:cross_domain` — Cross-domain generalisation | Untouched. |
| `sec:metrics` — Evaluation metrics | Replace metric-by-metric definitions with primary/secondary choice + bootstrap CI reporting; pointer to `\ref{sec:tb:metrics}`. |
| `sec:reproducibility` — Cost + reproducibility | Untouched. |
| `sec:interpretability` — Interpretability | Strip SHAP/Shapley-axiom paragraphs; keep project-specific decisions; pointer to `\ref{sec:tb:shap}`. |
| `sec:methodology_summary` — Summary | Light update to reflect new chapter. |

## Cross-reference updates

- `Overleaf/Chapters/5-Results.tex` line 26 (post-rename) currently points to
  `\ref{sec:metrics}` of Methodology; after split, formal metric definitions
  live in `\ref{sec:tb:metrics}`. Reword to point at both (applied choice in
  Methodology, formal definitions in new chapter), or re-point to the new
  chapter, whichever reads best.
- All other Methodology `\ref{}` from Results (lines 35, 142, 204, 579, 597)
  point at sections that *stay* in Methodology (tuning protocol, threshold-study
  protocol, leakage-free protocol, applied interpretability) and remain valid.
- The single Introduction reference (`sec:chapterMethodology`, line 68) remains
  valid.

## main.tex update

```latex
\input{Chapters/1-Introduction}
\input{Chapters/2-State of the Art}
\input{Chapters/3-Theoretical Background}     % new
\input{Chapters/4-Methodology}                 % was 3-
\input{Chapters/5-Results}                     % was 4-
\input{Chapters/6-conclusion}                  % was 5-
\input{Chapters/7-appendices}                  % was 6-
```

## Execution order

1. `git mv` the four chapter files to renumber (preserves history).
2. Update `Overleaf/main.tex` `\input{}` block.
3. Confirm renumbered tree compiles unchanged — isolates the rename from the
   content split.
4. Create the new `3-Theoretical Background.tex` file. Populate each section
   by moving (not paraphrasing) the relevant theory paragraphs from
   Methodology.
5. Edit `4-Methodology.tex` to remove migrated paragraphs and replace each
   with the appropriate `\ref{sec:tb:*}` pointer.
6. Re-audit attention-diagnostics location. Migrate if appropriate; drop
   section 8 of the new chapter if no separate theory block exists.
7. Update `\ref{sec:metrics}` cross-reference in Results.
8. Add missing glossary entries.
9. Update `.claude_context/PLAN.md` and `.claude_context/THESIS_STATE.md`.
10. Compile to verify zero new warnings, zero unresolved refs, correct TOC.
11. Commit + push (likely two commits: rename pass + content split).

## Verification

- LaTeX compile: zero new warnings, no undefined-reference warnings.
- TOC inspection in the generated PDF.
- `grep` for `\ref{sec:` across `Overleaf/Chapters/*.tex` and verify every
  target label exists.
- Read the new Methodology end-to-end and confirm no paragraph explains *what
  a tool is* — only *what we did with it*.
- `grep` for tool names (SMOTE, TreeSHAP, Tree-structured Parzen, Foggy Vision,
  …) and confirm each definition appears exactly once (in the new chapter).

## Out of scope

- **State of the Art** — unaffected.
- **Results, Conclusion, Appendices** — content unaffected; only filename
  renumbering and one possible `\ref{}` rewording.
- **Article 1 and Article 2** — entirely decoupled from this restructure.
  Both articles have their own self-contained Methodology / Experimental
  Framework sections and do not `\input{}` from `Overleaf/Chapters/`.
- **Experiments, numerical values, citations, BibTeX entries, figures,
  tables** — none touched.
