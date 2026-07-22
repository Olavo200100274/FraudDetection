# General/ Reference Curation Notes

> **Final project status (22 July 2026):** This is an archival record of the
> literature-curation stage. The dissertation is complete and the V3 PDF has
> been sent to the supervisors. The authoritative final selection is
> `Overleaf/references.bib`: it contains 46 entries, of which 41 are cited in
> the dissertation. Conditional references and Phase 5 writing instructions
> below are historical context rather than pending work.

**Curated by**: Claude Opus 4.7 (1M context)
**Date**: 2026-05-04
**Source**: triagem de `referencesForAproval/` (Phase 3 do plano de SoA)

The General/ folder holds **methodological / foundational** references used across the thesis (tabular DL, XAI, surveys, accuracy-vs-interpretability discussions). Dataset-specific references live in `BAF/` or `ULB/`.

---

## Tier 1 (cite directly in SoA)

| Citation key | Paper | File | Why |
|---|---|---|---|
| `huang2020tabtransformer` | Huang et al. (2020). *TabTransformer: Tabular Data Modeling Using Contextual Embeddings*. Amazon AWS, arXiv:2012.06678. | `General_Huang2020_TabTransformer.pdf` | Foundational tabular transformer using self-attention on categorical features. Direct ancestor / sibling of FT-Transformer. **Essential** for thesis SoA framing of tabular DL. |
| `somepalli2021saint` | Somepalli et al. (2021). *SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training*. Univ. Maryland + Capital One, arXiv:2106.01342. | `General_Somepalli2021_SAINT.pdf` | Tabular DL with self-attention + intersample attention. Reports outperforming XGBoost/LGBM/CatBoost on benchmarks. Strong contrast: in our work, FT-Transformer ties with CatBoost on BAF and loses on ULB — supports Maryam's framing "why simple LightGBM matches/beats tabular transformers". |
| `hilal2022financial` | Hilal, Gadsden, Yawney (2022). *Financial Fraud: A Review of Anomaly Detection Techniques and Recent Advances*. Expert Systems With Applications 193:116429. McMaster Univ. | `General_Hilal2022_AnomalyDetectionReview.pdf` | Q1 Elsevier review, comprehensive anomaly detection survey. Cite for "Fraud Detection in Financial Transactions" overview section. |
| `hernandezaros2024financial` | Hernandez Aros et al. (2024). *Financial fraud detection through ML: a literature review*. HSSC 11:1130. Springer Nature. | `General_HernandezAros2024_MLReview.pdf` | Recent (2024) PRISMA + Kitchenham SLR over 104 articles. Cite for ML fraud detection literature scope. |
| `ali2022financial` | Ali et al. (2022). *Financial Fraud Detection Based on ML: A SLR*. Applied Sciences 12:9637. MDPI. | `General_Ali2022_MLSLR.pdf` | SLR Kitchenham, 93 articles. Identifies SVM and ANN as popular ML approaches. Cite for ML fraud detection landscape. |
| `mienye2024deep` | Mienye & Jere (2024). *Deep Learning for CCFD: A Review of Algorithms, Challenges, and Solutions*. IEEE Access. | `General_Mienye2024_DLReview.pdf` | Recent review specifically on DL (CNN/RNN/LSTM/GRU) for credit card fraud. **Closest in spirit** to the comparative DL angle of our thesis. |
| `nesvijevskaia2021accuracy` | Nesvijevskaia et al. (2021). *The accuracy versus interpretability trade-off in fraud detection model*. Data & Policy 3:e12. Cambridge UP. | `General_Nesvijevskaia2021_AccuracyVsInterpretability.pdf` | **Directly aligned with thesis SHAP analysis + Maryam's "why simple LightGBM beats transformer" framing**. Discusses trade-off black-box ML vs interpretable models in fraud detection. |
| `molnar2022interpretable` | Molnar (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.). Independently published, online book. | `General_Molnar2022_InterpretableML.url` | Reference book for XAI. Cite for SHAP/LIME/permutation importance theory. |

**Total Tier 1 added to `references.bib`: 8 entries** (TabTransformer, SAINT, Hilal, Hernandez Aros, Ali, Mienye, Nesvijevskaia, Molnar).

---

## Tier 2 (cite if SoA includes matching subsection)

| Paper | File | Cite if... |
|---|---|---|
| Shenkar & Wolf (2022). *Anomaly Detection for Tabular Data with Internal Contrastive Learning*. ICLR 2022. | `General_Shenkar2022_AnomalyDetectionTabular.pdf` | SoA discusses unsupervised/anomaly detection on tabular (we have OCSVM as baseline). |
| Cartella et al. (2021). *Adversarial Attacks for Tabular Data: Application to Fraud Detection and Imbalanced Data*. Sony, arXiv:2101.08030. | `General_Cartella2021_AdversarialTabular.pdf` | SoA includes a "Threats to Validity" or "adversarial robustness" subsection. |
| Ti et al. (2022). *Feature generation and contribution comparison for electronic fraud detection*. Nature Sci Reports 12:18042. | `General_Ti2022_FeatureGeneration.pdf` | SoA discusses feature engineering / SHAP feature importance approaches. |
| Amarasinghe, Aponso, Krishnarajah (2018). *Critical Analysis of ML Based Approaches for Fraud Detection*. ACM. Univ. Westminster. | `General_Amarasinghe2018_CriticalAnalysis.pdf` | SoA includes a critical methodological retrospective. |

These are kept in `General/` but **not yet added to `references.bib`** — add them at SoA writing time only if the corresponding subsection materialises.

---

## Tier 3 (dropped from referencesForAproval — see `_dropped/`)

27 PDFs were dropped:
- 10 papers were already in `references.bib` under different filenames (e.g., the "1 How AI..." paper = `rymantubb2018survey`)
- 1 paper was an internal duplicate (`25 = 28` Mienye 2024)
- 16 papers were either off-scope (graph NN, federated learning, Fraud Triangle theory) or low-quality (predatory journals with implausible 97--100% accuracy claims, generic titles without specific contribution)

Full list with reasons in `referencesForAproval/_triage_notes.md`.

---

## Notes for Phase 5 (SoA writing)

When writing the SoA, the central narrative — per Maryam's directive — is:

> **"Why does a simple LightGBM outperform a sophisticated tabular transformer (FT-Transformer) on fraud detection?"**

The key citations to support this framing:
1. **`somepalli2021saint`** — claims SAINT (transformer) beats GBDT on benchmarks → contrast with our finding (FT-Transformer ≤ CatBoost on BAF/ULB)
2. **`huang2020tabtransformer`** — TabTransformer claims to "match" GBDT, not beat it → consistent with our finding
3. **`mienye2024deep`** — recent DL review shows mixed results
4. **`nesvijevskaia2021accuracy`** — interpretability vs accuracy trade-off framework
5. **`hilal2022financial`** + **`hernandezaros2024financial`** + **`ali2022financial`** — three recent surveys positioning the field
6. **`sun2025objective`** (already in `BAF/`) — directly states "objective over architecture" matters more, same dataset (BAF) and finding aligned with ours

This collection gives the SoA enough breadth to defend the central thesis question.
