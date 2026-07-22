# BAF Reference Curation Notes

> **Final project status (22 July 2026):** This is an archival record of the
> literature-curation stage. The dissertation is complete and the V3 PDF has
> been sent to the supervisors. The authoritative final selection is
> `Overleaf/references.bib`: it contains 46 entries, of which 41 are cited in
> the dissertation. Suggested searches, conditional additions, and other
> future actions below are historical notes and are not outstanding thesis
> tasks. No additional literature expansion is required for submission.

**Curated by**: Claude Opus 4.7 (1M context), in conversation with Olavo Caixeiro
**Date**: 2026-05-03
**Method**: Read first 10-12 pages of each PDF using PyMuPDF, classified against the 6 thesis contributions.

## Thesis contributions used as relevance criteria

1. **Unified comparison** classical ML vs FT-Transformer with leakage-free protocol
2. **Threshold selection** as a first-class experimental variable (4 strategies)
3. **Full factorial** 7 imbalance strategies × 5 models × 2 datasets
4. **Cross-domain generalization** Base→Variants I–V without retraining
5. **SHAP interpretability** with Jaccard cross-model and cross-variant
6. **FT-Transformer attention diagnostics** (Foggy Vision, entropy)

---

## Tier 1 — KEEP (cite directly in SoA)

| # | Citation key | Paper | Why keep |
|---|---|---|---|
| 1 | `jesus2022turning` | Jesus et al. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*. **NeurIPS 2022 Datasets and Benchmarks**. | Foundational BAF paper. Already in `.bib`. **Anchor citation** for any BAF discussion. |
| 2 | `cruz2023fairgbm` | Cruz et al. (2023). *FairGBM: Gradient Boosting with Fairness Constraints*. **ICLR 2023**. | Feedzai paper, uses BAF (called AOF). Operating point at 5% FPR — same as Jesus 2022. Aligns with thesis contribution 3 (factorial study) — provides fairness-constrained alternative within GBDT family. |
| 3 | `sun2025objective` | Sun et al. (2025). *Objective over Architecture: Fraud Detection Under Extreme Imbalance in Bank Account Opening*. **Computation 2025 (MDPI)**. | Same dataset (BAF Base 1M, 1.10% prevalence), compares LR/SVM/RF/LGBM/GRU. Their finding ("objective choice and operating point matter at least as much as architecture") **directly validates** thesis contribution 2 (threshold) and 1 (unified comparison). Their core features (`name_email_similarity`, `velocity_24h`) overlap with our SHAP findings. |
| 4 | `luzio2024decoupling` | Luzio et al. (2024). *Decoupling Decision-Making in Fraud Prevention through Classifier Calibration for Business Logic Action*. **ACM-SAC 2024**. | Uses BAF Base + all 5 Variants. Studies threshold/calibration decoupling — **closest existing work** to thesis contribution 2. Reports TPR@5%FPR (same as Jesus 2022). Different angle (calibration vs threshold strategies). |
| 5 | `alves2025openl2d` | Alves et al. (2025). *A benchmarking framework and dataset for learning to defer in human-AI decision-making*. **Nature Scientific Data 2025**. | Nature-published. Builds on BAF to create FiFAR L2D dataset. Cite as evidence of BAF ecosystem expansion. Subsumes the older `alves2023fifar` arXiv preprint. |

**Tier 1 BibTeX entries to add to `references.bib`: 4 new** (jesus2022 already present).

---

## Tier 2 — KEEP CONDITIONALLY (cite only if SoA has matching subsection)

| # | Citation key | Paper | Condition |
|---|---|---|---|
| 6 | `pombal2022understanding` | Pombal et al. (2022). *Understanding Unfairness in Fraud Detection through Model and Data Bias Interactions*. **KDD Workshop on ML in Finance 2022**. | KEEP if SoA has fairness/bias subsection. Otherwise drop — fairness is not a thesis contribution. |
| 7 | `nasif2026rhoss` | Nasif, Jahin, Mridha (2026). *Reinforcement-Guided Hyper-Heuristic Hyperparameter Optimization for Fair and Explainable SNN-Based Financial Fraud Detection*. **Knowledge-Based Systems**. | KEEP if SoA mentions exotic DL approaches on BAF. Q1 journal, BAF results (90.8% recall @ 5% FPR), includes XAI. Stronger than `perdigao2024neuromorphic`. **Use this OR perdigao, not both**. |
| 8 | `perdigao2024neuromorphic` | Perdigão et al. (2024). *Neuromorphic Bank Account Fraud Detection with Fairness-Aware Population Coding Mechanism*. **CISUC/Coimbra workshop**. | KEEP if SoA covers SNN approaches. Portuguese authors (Coimbra). Subsumed by `nasif2026rhoss` if we keep only one. |

**Tier 2 BibTeX entries to add: up to 3** (depending on final SoA structure).

---

## Tier 3 — DROP (move to `_dropped/`)

| # | Paper | Reason to drop |
|---|---|---|
| 9 | `BAF_FiFAR A Fraud Detection Dataset for Learning to Defer.pdf` (Alves et al., 2023, arXiv) | **Superseded** by `alves2025openl2d` (Nature 2025). Same authors, same dataset, more recent peer-reviewed version exists. |
| 10 | `BAF_Federated Learning for Financial Fraud Detection.pdf` (Lima, 2023, MSc thesis IST) | MSc thesis (not peer-reviewed paper). Federated learning topic outside thesis scope. |
| 11 | `BAF_SECURE AND SCALABLE HORIZONTAL FEDERATED.pdf` (Obiefuna et al., 2025, ICLR Workshop) | Federated learning topic outside thesis scope. ICLR Workshop is a credible venue but the topic doesn't fit. |
| 12 | `BAF_Online Banking ... Decentralized ML Framework.pdf` (AbouGrad & Sankuru, 2025, Mathematics MDPI) | Decentralized autoencoder + GDPR compliance focus. Outside thesis scope. |
| 13 | `BAF_A Deep Reinforcement Learning Framework ... Fraudulent Bank Account Openings.pdf` (Qayoom et al., 2024, SSURJET) | Reports **97% accuracy** on a 1.1% prevalence dataset — clear metric misuse (a constant negative classifier reaches 98.9% accuracy). Low-tier journal. **Could be cited as cautionary example** of poor evaluation, otherwise drop. |
| 14 | `BAF_A Unified Cryptographic and Machine Learning.pdf` (Muhammad, 2025, IJIRSET) | Doesn't use BAF (uses "mixed dataset of EU banking transactions and simulated breach logs"). Topic: post-quantum cryptography + federated GNN. Predatory journal characteristics. |
| 15 | `BAF_An Ensemble-based Fraud Detection ... Cyber Threat Classification.pdf` (Alhashmi et al., 2023, ETASR) | Reports 0.98 accuracy — same metric misuse as #13. Topic framed as cybersecurity rather than fraud detection methodology. |
| 16 | `BAF__Advanced Explainable Hybrid Metaheuristic-Deep Learning Framework ...pdf` (Madhu Kumar Reddy & Kiranbabu, 2026, IJACSA) | Reports 97.2% accuracy — same metric misuse. IJACSA = predatory journal. Poor writing quality (style errors). |
| 17 | `BAF_High-Recall Deep Learning A Gated Recurrent Unit Approach ...pdf` (Sun, Qi, Shen, 2025, ResearchSquare preprint) | **Redundant** with `sun2025objective` (same authors Sun & Shen, same dataset BAF Base, same finding GRU 78% recall / 5% precision). The Computation MDPI version is peer-reviewed and more complete; this preprint adds nothing. |

**Tier 3 actions**: 9 PDFs moved to `Overleaf/references/BAF/_dropped/`.

---

## MOVE — Misclassified

| # | Paper | Action |
|---|---|---|
| 17 | `BAF_Benchmarking Credit Card Fraud Detection Models A Comprehensive Evaluation Across Diverse Datasets.pdf` (Singh et al., 2025, Computational Economics Springer) | **Move to `Overleaf/references/ULB/`**. Uses ULB 2013, NOT BAF. Strong candidate (15 ML models, SMOTE/ADASYN, TabNet, SHAP) — relevant for ULB curation phase. |

---

## Notes (links — not papers)

| File | Purpose |
|---|---|
| `BAF_Bank Account Fraud (BAF) Tabular Dataset Suite - GitHub.txt` | Link to `github.com/feedzai/bank-account-fraud`. Cite as `\url{}` in dataset description, no `.bib` entry needed. |
| `BAF_Dataset Kaggle.txt` | Kaggle mirror link. Same as above. |

---

## Identified gaps (for future NotebookLM searches)

The current 18 BAF PDFs are **heavily concentrated** on:
- Fairness research (4 papers: Jesus, FairGBM, Pombal, Perdigão, Nasif)
- Federated learning (3 papers: drop tier)
- SNN/exotic DL (2 papers: Perdigão, Nasif)
- Predatory papers with metric misuse (3 papers: Qayoom, Alhashmi, Madhu Kumar)

**Missing on BAF specifically**:
- BAF + SMOTE/RUS/imbalance methodology study (only Jesus 2022 baseline)
- BAF + tabular transformers (TabTransformer, SAINT, FT-Transformer) — **this confirms thesis novelty**
- BAF + cross-domain Variants I–V deep study (only Jesus 2022 introduces them)
- BAF + SHAP interpretability — **this confirms thesis novelty**

**Suggested follow-up NotebookLM queries** (if needed):
- "Tabular transformers (FT-Transformer, TabTransformer, SAINT) on Bank Account Fraud"
- "Class imbalance resampling SMOTE evaluated on BAF benchmark"
- "Distribution shift evaluation on Bank Account Fraud Variants"
- "SHAP cross-model consistency on tabular fraud detection"

These gaps are **good news** for thesis positioning — they confirm the novelty of contributions 1 (unified ML vs FT-Transformer), 4 (cross-domain), and 5 (SHAP).

---

## Summary

- **Tier 1 keeps**: 5 papers (1 already in .bib, 4 new entries)
- **Tier 2 conditional**: 3 papers (depending on final SoA structure)
- **Tier 3 drops**: 9 papers (moved to `_dropped/`)
- **Moved to ULB/**: 1 paper (Singh 2025)
- **Notes**: 2 .txt link files (not citable)

**Net additions to `references.bib` after this curation**: 4 entries (Tier 1) + up to 3 (Tier 2 if needed).

**Total accounted for**: 5 (Tier 1) + 3 (Tier 2) + 9 (Tier 3) + 1 (moved) + 2 (notes) = **20 files** (matches the original BAF/ folder content).
