# PLAN.md — Sequential plan to thesis + article
**Last updated**: 2026-05-19  
**Current position**: ✅ Steps 1–8 COMPLETE (thesis first pass) → ✅ Step 8.1 COMPLETE (Maryam thesis-side feedback round 1: new "Theoretical Background" chapter; files renumbered 4–7) → ✅ Step 8.2 COMPLETE (Maryam thesis-side feedback round 2, item 1/3: short-form captions added to all 18 figures and 6 Appendix-B subfigures via `[short]{long}` so the List of Figures becomes concise; long captions preserved bit-for-bit; items 2/3 = List of Tables and 3/3 = page references still pending) → ✅ Steps 9–13.5 COMPLETE → ✅ Step 13.6 COMPLETE (Article 1 Maryam round-1 writing-only revision; Article 1 back in review queue) → ✅ Steps 15–19.6 COMPLETE on Article 2 → Article 1 awaiting round-2 Maryam review (Step 14); Article 2 awaiting round-1 Maryam review (Step 20)

---

## PART 1 — FINISH THE THESIS

### ✅ Step 1 — Reference curation (COMPLETE 2026-05-04/05)
- BAF curation (18 PDFs): 5 Tier 1 kept, 9 dropped → .bib +4 entries
- General Phase 3 (46 files from referencesForAproval): +9 Tier 1
- ULB curation (57 PDFs): 3 Tier 1 ULB + 4 Tier 1 General kept, 38 dropped → .bib +7 entries
- Foundational gap fill (10 PDFs): +10 critical entries (SMOTE, Vaswani, Shwartz-Ziv, etc.)
- **Final .bib: 41 entries. All 6 thesis contributions covered.**

### ✅ Step 2 — Decide and FREEZE SoA chapter structure (COMPLETE 2026-05-05)
Before writing a single line, commit to the section headings.  
Proposed structure (NOT yet frozen):
```
1. Fraud Detection in Financial Transactions (context + datasets)
2. Evaluation under Extreme Class Imbalance (PR-AUC, F-β, threshold, calibration)
3. Classical ML for Fraud Detection (LR, RF, GBDT families)
4. Deep Learning for Tabular Fraud Detection (TabTransformer, SAINT, FT-Transformer; DL vs GBDT debate)
5. Explainable AI in Fraud Detection (SHAP, LIME, accuracy-interpretability)
6. Synthesis: Limitations and Research Gaps (map 6 thesis contributions to gaps)
```
**Key: Section 4 is where Maryam's framing lives.** Must build to "why GBDT matches/beats Transformer."  
**Key: Section 6 is where thesis is "sold" — each gap maps to one contribution.**

### ✅ Step 3 — Write State of the Art chapter (COMPLETE 2026-05-05)
Section by section in order. Each section:
- Opens with function (what this covers and why)
- Synthesizes literature (argues, not enumerates)
- Identifies gap relevant to thesis
- Links to next section

**File**: `Overleaf/Chapters/2-State of the Art.tex`  
**Current state**: empty (only headings)

### ✅ Step 4 — Write Introduction chapter (COMPLETE 2026-05-05)
AFTER SoA — depends on it for the "motivation" section.
Structure: Context → Motivation → Research Questions (4-5 RQs mapping to 6 contributions) → Objectives → Document Structure

**File**: `Overleaf/Chapters/1-Introduction.tex`  
**Current state**: empty (only section headings)

### ✅ Step 5 — Write Conclusion chapter (COMPLETE 2026-05-06)
Structure: Summary of contributions → Answers to RQs → Limitations → Future Work

**File**: `Overleaf/Chapters/5-conclusion.tex`  
**Current state**: 1 sentence ("In this thesis we performed a comprehensive study of")

### ✅ Step 6 — Write Abstract EN + PT (COMPLETE 2026-05-06)
LAST — after all chapters done. ~250 words EN, careful PT translation.  
**Fix**: remove wrong keywords (drug discovery) from Abstract EN.

### ✅ Step 7 — Polish Appendices (COMPLETE 2026-05-06)
Decide: add real content (hyperparameter tables, bootstrap tables) or remove placeholders.  
**File**: `Overleaf/Chapters/6-appendices.tex`  
**Current state**: Lorem Ipsum

### ✅ Step 8 — Final review pass (COMPLETE 2026-05-10)
Verified during this pass:
- All `\ref{}` cross-references resolve; all `\cite{}` citations resolve (43 BibTeX entries)
- Notation consistent across chapters (PR-AUC, $\tau$, F$_2$, strategy names)
- 5 RQs in Introduction match 5 RQ headings in Conclusion verbatim
- 6 contributions enumerated in Conclusion match the 6 listed in THESIS_STATE.md

Concrete fixes applied:
- **Methodology** — added a new closing `Summary` section synthesising the protocol (replaced ending in SHAP one-hot detail)
- **Conclusion + Results** — corrected factual error: "RF the weakest in-domain model among supervised classifiers" was wrong (LR=0.143 < RF=0.159 on BAF); rephrased as "weakest of the high-capacity supervised classifiers" / "trailing the gradient-boosted models in-domain"
- **Conclusion + Results** — softened "Class Weights is consistently safe across all model families / never degrades substantially" — LGBM on ULB drops 17% under Class Weights, contradicting the absolute claim. Now framed as "safest default, with caveat for LGBM on ULB"

Threshold sensitivity tables (8 total) and SHAP Jaccard table audited and kept — each conveys complementary information not derivable from prose alone.

### ✅ Step 8.1 — Add Theoretical Background chapter (Maryam thesis-side round-1, COMPLETE 2026-05-19)

Maryam reviewed the thesis and observed that the Methodology chapter was mixing
two distinct kinds of content: textbook-style theory (what each tool *is* and
how it works in general) and project-specific application (how *this* study
configured those tools). She asked for a structural split: a new chapter
between State of the Art and Methodology to hold the toolkit definitions, so
that the Methodology can focus exclusively on the project's protocol and
cross-reference the new chapter when background is needed.

User decisions in this round:
- **Chapter name**: *Theoretical Background*.
- **File numbering**: renumber all subsequent chapter files so filename index
  matches chapter order.

Concrete changes:
1. **Renumbered chapter files (git-mv, history preserved)** in commit `c82384e`:
   `3-Methodology.tex → 4-Methodology.tex`,
   `4-Results.tex → 5-Results.tex`,
   `5-conclusion.tex → 6-conclusion.tex`,
   `6-appendices.tex → 7-appendices.tex`.
   `Overleaf/main.tex` `\input{}` block updated to include the new slot.
2. **Created** `Overleaf/Chapters/3-Theoretical Background.tex` (label
   `sec:chapterTheoreticalBackground`) with seven sections that mirror, in
   order, the tool families invoked by Methodology:
   `sec:tb:models` (LR, RF, LightGBM, CatBoost, OCSVM, FT-Transformer);
   `sec:tb:imbalance` (RUS, ROS, SMOTE family, Class Weights);
   `sec:tb:metrics` (Precision, Recall, $F_\beta$, PR-AUC vs.\ ROC-AUC, calibration);
   `sec:tb:thresholds` (fixed default, validation-optimal $F_\beta$, precision-constrained, alert-budget);
   `sec:tb:hpo` (SMBO, TPE algorithm, pruning);
   `sec:tb:shap` (Shapley values, axioms, TreeSHAP, LinearExplainer, GradientExplainer, Jaccard agreement).
   No new citations introduced beyond those already in `references.bib`.
   Attention-diagnostics theory was *not* migrated because it currently lives
   in Results §7 (Article 2's owned content), not in Methodology — so the
   originally-planned 8th section was dropped.
3. **Trimmed** `Overleaf/Chapters/4-Methodology.tex`: stripped theoretical
   passages from six sections (Models, Imbalance Strategies, Threshold Study,
   Hyperparameter Tuning, Evaluation Metrics, Interpretability) and replaced
   each with a one-paragraph project-specific statement plus a pointer to the
   corresponding `\ref{sec:tb:*}` label. Existing Methodology labels
   unchanged; existing TikZ figures and the leakage-free protocol untouched.
   The three remaining `enumerate` blocks (contributions C1–C4, leakage-free
   CV steps, cross-domain protocol steps) are project-specific procedural
   lists and were kept as-is.
4. **Updated** `Overleaf/Chapters/5-Results.tex` line 26 chapter-intro
   cross-reference: formal metric definitions now point to
   `\ref{sec:tb:metrics}` in the new chapter; the applied metric reporting
   choices still point to `\ref{sec:metrics}` in Methodology.
5. **Updated** `Overleaf/Chapters/1-Introduction.tex` Document Structure
   section: added a paragraph describing the new Theoretical Background
   chapter; rephrased the Methodology paragraph to reflect that the chapter
   now describes "how the toolkit was applied" rather than restating each
   tool.
6. **No changes** to State of the Art, references.bib, glossary
   (acronyms already cover all introduced terms), Conclusion, Appendices, or
   either Article.

Cross-reference health checked: every `\ref{sec:tb:*}` in Methodology and
Results resolves to a label defined in `3-Theoretical Background.tex`; the
single SoA cross-reference added in Methodology
(`\ref{subsec:sota-dl-gbdt-debate}`) matches the existing SoA label.

### ✅ Step 8.2 — Short caption forms for the List of Figures (Maryam thesis-side round 2, item 1/3, COMPLETE 2026-05-19)

Maryam pointed out that the List of Figures was cluttered because every
entry was the full long caption. She asked for two things per figure:
(a) confirmation that the long caption text is still appropriate; and
(b) addition of a `[short]{long}` LaTeX optional argument so the LoF entry
becomes concise while the in-page caption stays detailed.

User decisions in this round:
- Subfigures in Appendix B also receive short forms.

A Phase 1 audit covered every figure in `Overleaf/Chapters/*.tex` (no
figures in Introduction, SoA, Theoretical Background, or Conclusion). All
18 long captions were assessed as KEEP_AS_IS and preserved bit-for-bit.
Twenty-four `[short]` arguments were added (3 in Methodology + 13 in
Results + 2 outer + 6 subcaptions in Appendix B). Style: 6–10 words per
short form; series parallelism for the four ΔPR-AUC/ΔF2 heatmaps
(§5.3–§5.6) and for the FP/FN subfigure pairs in Appendix B.

The plan record is preserved verbatim at
`.claude_context/FiguresShortCaptionsPlan.md` for future reference.

Items still pending in this round:
- Item 2/3 — List of Tables (apply same `[short]{long}` treatment to the
  thesis's 30 tables, with the same audit-then-add pattern).
- Item 3/3 — page references where appropriate (still to be scoped: where
  in the thesis does explicit `Section X (p. NN)`-style cross-referencing
  add real value, and what is the LaTeX mechanism — `\pageref{}` or the
  `hyperref`/`varioref` family).

---

## PART 2 — PAPER 1: BENCHMARK + THRESHOLD

**Maryam directive (revised 2026-05-07):** the original single-TPAMI-article plan is replaced by **two separate papers**, splitting the thesis Results by topic and audience. Paper 1 is the practitioner-oriented "What" paper centred on threshold sensitivity. Folder: `Article 1/` (Elsevier CAS template).

### Paper 1 — Section ownership
- Results §1 (Baseline model comparison)
- Results §2 (Threshold sensitivity analysis) ⭐ **centrepiece**
- Results §3 (Full factorial imbalance × model)
- Bootstrap CIs + computational cost tables
- Decision framework table

### Paper 1 — Main claims
1. Threshold choice can swing F₂ by an order of magnitude (9× on BAF), dwarfing model and strategy effects
2. No imbalance strategy consistently dominates; Class Weights is the safest default
3. SMOTE+Tomek = SMOTE exactly under extreme imbalance (Tomek removes nothing)
4. Actionable decision framework for practitioners

### ✅ Step 9 — Choose venue + freeze title (COMPLETE 2026-05-10)
- **Venue**: Expert Systems with Applications (ESWA, Elsevier, Scopus Q1) — applied, practitioner-oriented, table-heavy ✅
- **Title**: *"Threshold Selection as a Critical Design Choice in Financial Fraud Detection: A Systematic Comparison of Classical Machine Learning Models and Tabular Transformers"*
- **Short title**: *"Threshold Selection in Financial Fraud Detection"*
- **Template**: `Article 1/main.tex` (Elsevier CAS single-column; renamed from `cas-sc-template.tex` by the user on Overleaf) ✅
- **Bibliography**: `Article 1/cas-refs.bib` replaced with thesis `Overleaf/references.bib` (43 entries; +4 added in Step 13.5 for tibshirani1996regression, breiman2001random, scholkopf2001estimating, fernandez2018smote) ✅

### ✅ Step 10 — Draft Method (COMPLETE 2026-05-10)
Wrote §3 Experimental Framework: datasets, 6 models, 7 strategies, 4 threshold strategies, leakage-free protocol, Optuna HPO, metrics. Cross-domain, SHAP, attention excluded.

### ✅ Step 11 — Draft Results (COMPLETE 2026-05-10)
All tables written: baseline ULB + BAF (Tables 2–3), threshold sensitivity F₂ ULB + BAF + Alert Rate (Tables 4–6, centrepiece), full factorial PR-AUC + F₂ (Tables 7–9), bootstrap CIs + computational cost (Tables 10–11).

### ✅ Step 12 — Draft Discussion + Decision Framework (COMPLETE 2026-05-10)
§5 Discussion with 4 subsections (threshold dominates; no strategy universally dominates; architecture-specific SMOTE sensitivity; decision framework Table 12). Companion paper pointer in §5.3.

### ✅ Step 13 — Draft Introduction + Related Work + Conclusion + Abstract (COMPLETE 2026-05-10)
§1 Introduction (4 contributions C1–C4), §2 Related Work (3 subsections, ~12 refs), §6 Conclusion (4 paragraphs), Abstract (~250 words), 5 highlights, 7 keywords. File: `Article 1/cas-sc-template.tex` (renamed to `main.tex` on Overleaf in Step 13.5).

### ✅ Step 13.5 — Post-draft polish (COMPLETE 2026-05-11)
Polish work driven by Overleaf review of the compiled draft and by feedback on visual layout/text flow. Spans both the thesis and Paper 1.

**Article 1 polish:**
- **Institutional metadata corrected** — affiliation changed from IPCB/Castelo Branco (placeholder) to *Departamento de Informática e Métodos Quantitativos, Escola Superior de Gestão e Tecnologia, Instituto Politécnico de Santarém*; correct emails (`@esg.ipsantarem.pt`); real ORCIDs for Olavo, Maryam, Pedro.
- **Missing references added** — `tibshirani1996regression`, `breiman2001random`, `scholkopf2001estimating`, `fernandez2018smote` added to `cas-refs.bib` (resolved 4 undefined citation warnings).
- **Broken cross-reference fixed** — `\ref{tab:ops_baf}` (label never defined) corrected to `\ref{tab:baseline_baf}`.
- **Numerical correction** — recall range "43–47%" → "41–47%" (RF recall = 0.415, not 0.43).
- **Float placement fix (CAS template)** — root cause: `cas-sc.cls` loads `stfloats` and `cas-common.sty` redefines `table`/`figure` internally via `\@float{}`, ignoring user placement options (`[H]`, `[ht!]`, `\FloatBarrier` all silently dropped). Solution: convert every `table`/`figure` environment to a `minipage` + `\captionof` (from `capt-of`), which is *not* a float and therefore appears exactly where placed in source. Applied to all 12 tables and 2 figures in `Article 1/main.tex`.
- **Template scaffolding cleanup** — removed `cas-grabs.pdf`, `cas-munnar-2024.jpg`, `cas-pic1.pdf` placeholder files. `cas-sc-template.tex` deleted (replaced by `main.tex` on Overleaf side).

**Thesis polish:**
- **Float placement fix (thesis class)** — same minipage approach applied to `Overleaf/Chapters/3-Methodology.tex` (3 TikZ figures), `Overleaf/Chapters/4-Results.tex` (30 tables + 13 figures), and `Overleaf/Chapters/6-appendices.tex` (2 tables). Appendix B's 2 figures with `subfigure` content reverted to `figure[H]` because `subcaption` errors out outside a real float; `[H]` (from `float`, already in preamble) gives exact placement.
- **Caption warning suppression** — added `\captionsetup{hypcap=false}` to `Overleaf/include/preamble.tex`; eliminates the "hypcap=true will be ignored" warning that `\captionof` triggers inside minipages (~12 warnings → 0).
- **Results chapter prose revision (4-Results.tex)** — full pass: (1) removed every `\paragraph{Discussion.}` block (8 in total) and integrated their content into the surrounding narrative as flowing prose; (2) added a chapter-level overview paragraph that explains how the section order builds; (3) added lead-in sentences before each table/figure ("Tables X--Y report..." / "Figure X visualises...") and interpretive sentences after; (4) condensed §9 from "Discussion" (6 subsections, heavy duplication of per-section discussions) into "Synthesis and Threats to Validity" (one synthesis paragraph + 5-item validity list). File length: 1045 → 841 lines, with all 30 tables, 13 figures, 43 labels and 42 internal `\ref{}` preserved bit-for-bit.

### ✅ Step 13.6 — Maryam round-1 writing-only revision pass (COMPLETE 2026-05-19)
Maryam reviewed the first Article 1 draft and returned a single high-level note:
*"it has too much itemize, it is strange!"*. She also authored a detailed revision
prompt (preserved verbatim in `.claude_context/RevisingPaperContext.md`) covering:
reducing thesis-like writing, removing excessive enumeration/itemize, deduplicating the
four most-repeated ideas (fixed $\tau{=}0.5$ fails, threshold dominates,
SMOTE = SMOTE+Tomek, threshold is first-order), replacing dramatic wording
("catastrophic"/"destroying recall"/"detect nothing") with journal-grade equivalents,
strengthening mechanistic interpretation in the Discussion, reframing the paper
around score-space behaviour / probability compression / operating-point instability
rather than the bare slogan "threshold tuning matters", tightening Introduction and
Conclusion, and smoothing inter-subsection transitions.

Applied the prompt as a pure writing revision (no experiments, numbers, tables,
citations, figures, or labels touched). Concrete edits in the new
`Article 1/main.tex`:
- Models / Class Imbalance Strategies / Threshold Strategies sections converted from
  `\begin{itemize}`/`\begin{enumerate}` lists to integrated prose paragraphs.
- Introduction contributions block (C1–C4) converted from `enumerate` to a single
  flowing paragraph.
- Discussion §5.1 retitled "Score Geometry under Extreme Imbalance" (was "Threshold
  Selection Dominates Model Selection") and rewritten around the
  probability-compression mechanism rather than as a restatement of the threshold
  tables.
- Conclusion rewritten to open with the score-geometry message and discuss broader
  implications, instead of repeating the BAF/ULB numerical findings already given in
  Results and Discussion.
- Dramatic wording replaced throughout ("catastrophic" → "operationally unsuitable";
  "destroying recall" → "near-collapse of recall"; "detect nothing" → "flags almost
  nothing"). The four over-repeated ideas now each appear with venue-grade rephrasing
  rather than near-identical restatement across Intro/Results/Discussion/Conclusion.

File policy: original draft preserved as `Article 1/main_old.tex`; revised draft is
now `Article 1/main.tex` so Overleaf compiles the new version by default. The lone
remaining itemize is the Highlights block, which is required by the ESWA template.

### Step 14 — Maryam review → iterate → submit

---

## PART 3 — PAPER 2: MECHANISTIC / ROBUSTNESS

**Maryam directive (revised 2026-05-07):** the "Why" paper that explains mechanistically why GBDTs match/beat tabular transformers on fraud data. Folder: `Article 2/` (Elsevier CAS template — may swap to IEEEtran if TNNLS is chosen).

### Paper 2 — Section ownership
- Results §4 (Transformer robustness to imbalance strategies)
- Results §5 (Cross-domain generalisation)
- Results §6 (SHAP interpretability)
- Results §7 (FT-Transformer attention diagnostics) ⭐ **centrepiece**
- Discussion on classical ML vs transformer

### Paper 2 — Main claims
1. SMOTE degrades FT-Transformer PR-AUC by 41% vs only 14% for CatBoost — mechanistically explained via Foggy Vision attention
2. RF is the most cross-domain robust model despite lower in-domain scores
3. LGBM SHAP feature ranking is perfectly stable across all 6 BAF variants (Jaccard=1.00), explaining why interpretability holds under shift
4. Top attention tokens align with top SHAP features across architecturally different models — **convergent validity**

### ✅ Step 15 — Choose venue + freeze title (REVISED 2026-05-14)
- **Venue** (final): **Applied Soft Computing (Elsevier, Scopus Q1)** — better fit than Neural Networks for applied+mechanistic empirical work; CAS template retained, no swap
- **Title** (frozen 2026-05-14): *"Why Simple Tree-Based Models Suffice for Financial Fraud Detection: Mechanistic Evidence from SHAP–Attention Convergence and Cross-Domain Transfer"*
- **Short title**: *"Why Simple Trees Suffice for Financial Fraud Detection"*
- **Template**: `Article 2/main.tex` (Elsevier CAS single-column) ✅
- **Bibliography**: `Article 2/cas-refs.bib` (copied from Article 1's bib, +2 entries: `grinsztajn2022tree`, `caixeiro2025threshold` placeholder) ✅
- **Earlier interim titles** (rejected during 2026-05-14 review): "Mechanistic Understanding of Tabular Transformer Limitations..." (too transformer-centric) and the original plan's "Why LightGBM Generalises Better Than Tabular Transformers..." (factually weak — LGBM does not literally beat FT-T cross-domain; RF is the real cross-domain winner)

### ✅ Step 16 — Draft Method (COMPLETE 2026-05-13)
Wrote §3 Experimental Setup: datasets (BAF suite description, ULB role), models, cross-domain transfer protocol (zero-shot Base→Variants I-V), SHAP analysis setup (3 explainers, Jaccard analysis), attention diagnostic framework (4 patterns + entropy). References companion Paper 1 for the shared leakage-free protocol.

### ✅ Step 17 — Draft Results (COMPLETE 2026-05-13)
§4 Results with 4 subsections:
- §4.1 Imbalance sensitivity (Tables 1-2: FT-T vs CatBoost on ULB + BAF)
- §4.2 Cross-domain robustness (Tables 3-5 + **new Figure 1: cross-domain PR-AUC line chart** showing RF improving on 4/5 variants)
- §4.3 SHAP attribution (Figure 2: shap_global, Figure 3: shap_beeswarm, Tables 6-7: Jaccard cross-model + cross-variant, Figure 4: shap_waterfall)
- §4.4 Attention diagnostics (Figure 5: attention_aggregate, Figure 6: attention_heatmap_fp, Foggy Vision diagnosis)

Figures: 6 total. New figure generated: `Article 2/figs/crossdomain_prauc.pdf` (via `src/generate_crossdomain_chart.py`).

### ✅ Step 18 — Draft Discussion + Convergent Validity (COMPLETE 2026-05-13)
§5 Discussion with 4 subsections:
- §5.1 SMOTE-FT-T interaction: synthetic interpolates distort joint feature distribution seen by attention; tree splits less sensitive
- §5.2 RF cross-domain regularization: random feature subsampling prevents domain-specific co-occurrence overfitting
- §5.3 SHAP stability as diagnostic: Jaccard=1.00 → degradation is in decision surface, not in which features are informative
- §5.4 Convergent validity + Foggy Vision: H=0.976 confirms BAF signal ceiling; attention agrees with SHAP on top-3 features

### ✅ Step 19 — Draft Introduction + Related Work + Conclusion + Abstract (COMPLETE 2026-05-13)
§1 Introduction (4 contributions C1-C4), §2 Related Work (3 subsections: tabular DL debate, distribution shift, SHAP + attention), §6 Conclusion (4 paragraphs), Abstract (~270 words), 5 highlights, 8 keywords.

### ✅ Step 19.5 — Reframe to "Simple Trees Suffice" + venue switch + figure overhaul (COMPLETE 2026-05-14)
Critical-review pass driven by user concern that the original draft framing did not align with Maryam's directive *and* that the data did not literally support the planned LGBM-centric framing. Three structural changes:

1. **Title and framing**: shifted from "Tabular Transformer Limitations" to **"Why Simple Tree-Based Models Suffice for Financial Fraud Detection: Mechanistic Evidence from SHAP–Attention Convergence and Cross-Domain Transfer"**. Reframed Abstract, §1 hook + 4 contributions (C1–C4 now framed as evidence FOR sufficiency), Highlights, Keywords (convergent validity now front-loaded), §6 Conclusion (opens with 5-pillar argument).
2. **Content expansion** (~6.5K → ~8.8K words): expanded §2 Related Work (3 deeper subsections: tabular DL with grinsztajn structural analysis + fraud-specific factors; cross-domain bias–variance theory; convergent validity methodology), expanded §5.1–§5.4 (tokenisation analysis under SMOTE; bias–variance under shift; regulator/audit implications of SHAP stability; "when does attention help" + Foggy Vision as useful negative result), added **NEW §5.5 "Why Simple Trees Suffice: The Combined Argument"** with 5 explicit pillars (no resampling penalty, cross-domain parity/advantage, stable explanations, no headroom above ceiling, 40× cheaper compute).
3. **Figure overhaul** (6 → 8 main figures): removed `shap_dependence.pdf` (no clear narrative); generated 2 new figures via Python scripts:
   - `src/generate_smote_asymmetry_chart.py` → `Article 2/figs/smote_asymmetry.pdf` (FT-T vs CatBoost across 7 strategies × 2 datasets, visualises the 41/14 asymmetry)
   - `src/generate_convergent_validity_chart.py` → `Article 2/figs/convergent_validity.pdf` (LGBM SHAP top-10 vs FT-T attention top-10, with 5 overlapping features highlighted in green; top-2 identical: `device_os`, `housing_status`)
4. **Venue switch**: Neural Networks → **Applied Soft Computing** (better fit for applied+mechanistic empirical work; same CAS template).

Files touched: `Article 2/main.tex`, `src/generate_smote_asymmetry_chart.py` (new), `src/generate_convergent_validity_chart.py` (new), `Article 2/figs/smote_asymmetry.pdf` (new), `Article 2/figs/convergent_validity.pdf` (new), `Article 2/figs/shap_dependence.pdf` (deleted), `.claude_context/PLAN.md`, `memory/project_overview.md`.

### ✅ Step 19.6 — Article 2 deep-review pass (COMPLETE 2026-05-14, commit `6c25574`)
Systematic verification of the Step 19.5 draft against actual experimental data caught **7 factual inconsistencies** that a reviewer would have spotted:

1. **§4.4 line 793-798**: "Top-3 attention tokens *precisely* match the top-3 tree-based SHAP features" was FALSE. Only top-2 coincide (`device_os`, `housing_status`); the 3rd diverges (`employment_status` attention vs `phone_home_valid` SHAP). Reworded to *"the first two are also the top two SHAP features ... the third (`employment_status`) sits inside the LGBM SHAP top-15"*.
2. **§5.4**: convergent validity "agree on the same three primary signals (`device_os`, `housing_status`, `phone_home_valid`)" was FALSE (`phone_home_valid` is rank 7 in attention, not 3). Reframed to *"agree on the top two ... and overlap on five of ten"*, with explicit cross-model + cross-method framing (stronger, not weaker, claim).
3. **§6 Conclusion**: methodological contribution said convergent validity was applied "on the **same** fraud detection model" — FALSE: LGBM (SHAP) vs FT-Transformer (Attention) are different models. Reworded to make doubly cross-method framing explicit.
4. **§3.1 BAF Variants**: per-variant descriptions ("Variant I: bias injection...", "Variant IV: feature quality under adversarial conditions") were invented and don't match the BAF paper. Replaced with the authors' own taxonomy (group-size, prevalence, separability, training bias) and a deferral to `jesus2022turning`.
5. **§4.2 / §5.2 / §5.3**: prose tying specific variants to "covariate" vs "label" shift — generalised to "distributional shifts" of varying severity.
6. **§4.2 F₂ claim**: "RF remains above its Base score on Variants I, III, and IV" was FALSE for Variant III (0.289 < 0.294). Corrected to "above on I (0.307) and IV (0.314), tying on III (within noise), dropping clearly on V (0.278)".
7. **Compute claim**: "~40× training-time advantage" was inflated. Actual values: 346.9/16.1 = **21.5× training**, 46,550/1,243 = **37.4× tuning**. Corrected in Abstract, §1, §5.5 Pillar 5, and §6 (v) to "~20× training, ~37× tuning".

Verified after fixes: 22/22 citations resolve in `cas-refs.bib`; 5 pillars consistent across Abstract / §5.5 / §6; word count 8783 → 8999 (within plan target).

### Step 20 — Maryam review → iterate → submit

---

## How to update this file

When a step is completed, change its bullet to ✅ and update "Current position" header.  
When starting a new step, add date and specific decisions made (e.g., which SoA structure was frozen in Step 2).
