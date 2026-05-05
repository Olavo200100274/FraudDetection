# PLAN.md — Sequential plan to thesis + article
**Last updated**: 2026-05-05  
**Current position**: ✅ Steps 1–3 COMPLETE → entering Step 4 (Introduction)

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

### Step 4 — Write Introduction chapter
AFTER SoA — depends on it for the "motivation" section.
Structure: Context → Motivation → Research Questions (4-5 RQs mapping to 6 contributions) → Objectives → Document Structure

**File**: `Overleaf/Chapters/1-Introduction.tex`  
**Current state**: empty (only section headings)

### Step 5 — Write Conclusion chapter
Structure: Summary of contributions → Answers to RQs → Limitations → Future Work

**File**: `Overleaf/Chapters/5-conclusion.tex`  
**Current state**: 1 sentence ("In this thesis we performed a comprehensive study of")

### Step 6 — Write Abstract EN + PT
LAST — after all chapters done. ~250 words EN, careful PT translation.  
**Fix**: remove wrong keywords (drug discovery) from Abstract EN.

### Step 7 — Polish Appendices
Decide: add real content (hyperparameter tables, bootstrap tables) or remove placeholders.  
**File**: `Overleaf/Chapters/6-appendices.tex`  
**Current state**: Lorem Ipsum

### Step 8 — Final review pass (BOTH Methodology AND Results)
- Check Results for redundant tables/figures that "raise more questions than they answer" → remove them
- Cross-references and citations all resolve
- Notation consistency across chapters
- LaTeX compiles without new warnings
- Narrative coherence: read Introduction→Conclusion as a single story

---

## PART 2 — WRITE THE ARTICLE

### Step 9 — Decide title + tone (TPAMI)
- Venue: **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**
- Tone: rigorous, measured (NOT "Tabular DL is not all you need" style — too informal for TPAMI)
- Title direction: "A Leakage-Free Comparative Study of Classical Machine Learning and Tabular Transformers for Financial Fraud Detection"
- Length: ~12-14 pages (TPAMI standard)

### Step 10 — Draft Method section
Compact from thesis Methodology (~2 pages). Keep: leakage-free protocol, 6 models, 7 strategies, 4 threshold strategies, factorial, cross-domain, SHAP, attention. Cut: all implementation detail.

### Step 11 — Draft Results section
Compact from thesis Results (~4 pages). Key tables/figures: baseline PR-AUC, factorial heatmaps, cross-domain, SHAP Jaccard, attention diagnostic. Cut: operational metrics, verbose discussion.

### Step 12 — Draft Related Work
Compact from thesis SoA (~1.5 pages). Focus: DL vs GBDT tabular + threshold + leakage. ~15-20 refs.

### Step 13 — Draft Discussion
**Heaviest section for TPAMI**. Argue carefully why GBDT matches/beats FT-Transformer:
1. Calibration: GBDT produces well-calibrated probabilities natively
2. Imbalance robustness: FT-Transformer SMOTE degradation 41% vs 14% CatBoost
3. Architectural mismatch: Transformer designed for sequences; tabular ≠ sequential
4. Dataset ceiling: Foggy Vision confirms 0.18 PR-AUC is dataset-limited, not model-limited

### Step 14 — Draft Introduction + Conclusion + Abstract
After the other sections — easier to write last.

### Step 15 — Review with Maryam
Send full draft for review.

### Step 16 — Iterate + Submit to TPAMI

---

## How to update this file

When a step is completed, change its bullet to ✅ and update "Current position" header.  
When starting a new step, add date and specific decisions made (e.g., which SoA structure was frozen in Step 2).
