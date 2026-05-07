# PLAN.md — Sequential plan to thesis + article
**Last updated**: 2026-05-06  
**Current position**: ✅ Steps 1–7 COMPLETE → entering Step 8 (Final review pass)

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

### Step 8 — Final review pass (BOTH Methodology AND Results)
- Check Results for redundant tables/figures that "raise more questions than they answer" → remove them
- Cross-references and citations all resolve
- Notation consistency across chapters
- LaTeX compiles without new warnings
- Narrative coherence: read Introduction→Conclusion as a single story

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

### Step 9 — Choose venue + freeze title
- Candidates: **Expert Systems with Applications**, Decision Support Systems, Information Sciences (all Elsevier)
- Title direction: *"Threshold Selection as a Critical Design Choice in **Financial** Fraud Detection: A Systematic Comparison of Classical ML and Tabular Transformers"* (broaden from "Credit Card" — thesis covers BOTH ULB and BAF; the 9× threshold finding is empirically strongest on BAF)
- Confirm CAS template is appropriate for chosen venue
- Length target: 10,000–12,000 words, table-heavy

### Step 10 — Draft Method (compact ~2 pages)
Keep: leakage-free protocol, 6 models, 7 strategies, 4 threshold strategies. Cut: cross-domain, SHAP, attention (those belong to Paper 2).

### Step 11 — Draft Results (table-heavy)
Baseline PR-AUC tables (ULB + BAF), threshold sensitivity tables (centrepiece), full factorial 7×5×2 tables, bootstrap CIs, computational cost tables.

### Step 12 — Draft Discussion + Decision Framework
The 4 main claims above. Decision framework table (recommended configuration by deployment scenario) is the headline practitioner deliverable.

### Step 13 — Draft Introduction + Related Work + Conclusion + Abstract
Related Work focuses on threshold selection + class imbalance + leakage in fraud detection (~10–15 refs). Less emphasis on tabular-DL debate (that's Paper 2's territory).

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

### Step 15 — Choose venue + freeze title
- Candidates: **IEEE TNNLS** (IEEEtran template), Neural Networks (Elsevier), Applied Soft Computing (Elsevier)
- Title direction: *"Why LightGBM Generalises Better Than Tabular Transformers Under Distribution Shift: Cross-Domain Evidence and Attention Diagnostics in Fraud Detection"*
- Swap template if TNNLS chosen; otherwise CAS template stands
- Length target: 9,000–11,000 words, figure-heavy

### Step 16 — Draft Method (compact, references Paper 1 for shared protocol)
Cross-domain transfer protocol, SHAP setup (LinearExplainer / TreeExplainer / GradientExplainer), attention diagnostic framework (4 patterns + entropy). Reference Paper 1 for the leakage-free protocol details to avoid duplication.

### Step 17 — Draft Results (figure-heavy)
Transformer-vs-tree under strategies (Tables), cross-domain Base→Variants (Tables + figures), SHAP cross-model + cross-variant Jaccard (Tables), attention aggregate distribution + heatmaps (centrepiece figures).

### Step 18 — Draft Discussion + Convergent Validity
The 4 mechanistic arguments + convergent-validity claim (attention top tokens ≈ SHAP top features) + Foggy Vision interpretation as dataset ceiling rather than model failure.

### Step 19 — Draft Introduction + Related Work + Conclusion + Abstract
Related Work focuses on tabular DL debate, attention interpretability, distribution shift in fraud (~15–20 refs).

### Step 20 — Maryam review → iterate → submit

---

## How to update this file

When a step is completed, change its bullet to ✅ and update "Current position" header.  
When starting a new step, add date and specific decisions made (e.g., which SoA structure was frozen in Step 2).
