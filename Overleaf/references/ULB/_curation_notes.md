# ULB/ Reference Curation Notes

**Curated by**: Claude Opus 4.7 (1M context)
**Date**: 2026-05-04 (Phase 2 part 1; second half pending)
**Source**: User uploaded ~57 PDFs to ULB/ during Phase 1-3.

The ULB/ folder holds references that **use the ULB Credit Card Fraud Detection 2013 dataset** (Dal Pozzolo et al., 284K transactions, 0.17% fraud, PCA features V1-V28). Methodologically broad references that don't tie to ULB specifically go to General/.

---

## Phase 2 PART 1 — First half (29 PDFs triaged)

### Tier 1 — KEEP (cite directly in SoA)

| Citation key | Paper | File | Why |
|---|---|---|---|
| `dalpozzolo2015calibrating` | Dal Pozzolo et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI. | `ULB_DalPozzolo2015_CalibratingUndersampling.pdf` | Foundational ULB paper. Already curated from referencesForAproval. |
| `leevy2023binary` | Leevy, Hancock, Khoshgoftaar (2023). *Comparative analysis of binary and one-class classification techniques for credit card fraud data*. Journal of Big Data (Open Access). | `ULB_Leevy2023_BinaryVsOneClass.pdf` | **Compares 5 BCC vs 3 OCC learners on ULB. Top binary classifier: CatBoost.** Directly supports our OCSVM-as-weak-baseline finding. |
| `singh2025benchmarking` | Singh, Singh, Kumar (2025). *Benchmarking CCFD Models: A Comprehensive Evaluation Across Diverse Datasets*. Computational Economics (Springer). | `ULB_Singh2025_BenchmarkingCCFD.pdf` | 15 ML algorithms incl. TabNet (tabular DL); SMOTE+ADASYN; SHAP. Uses ULB 2013 + synthetic 2019 + real 2023. **Strongly aligned with thesis framing** (classical ML vs tabular DL on fraud, with XAI). |

### Tier 1 — Moved to General/ (broader methodological scope, not ULB-specific)

| Citation key | Paper | New filename | Why moved |
|---|---|---|---|
| `thimonier2024comparative` | Thimonier et al. (2024). *Comparative Evaluation of Anomaly Detection Methods for Fraud Detection in Online Credit Card Payments*. Univ. Paris-Saclay. | `General/General_Thimonier2024_AnomalyDetectionEvaluation.pdf` | **Finding: "LightGBM exhibits significantly superior performance" over anomaly detection methods.** Directly supports Maryam's "why simple LightGBM beats sophisticated approaches" framing. Uses real online credit card data (likely ULB but methodology is general). |
| `imani2026why` | Imani et al. (2026). *Why ROC-AUC Is Misleading for Highly Imbalanced Data*. MDPI. | `General/General_Imani2026_WhyROCAUCMisleading.pdf` | Uses ULB as one of 3 imbalanced benchmarks but the contribution is **methodological evaluation** (PR-AUC > ROC-AUC under extreme imbalance) — applies to fraud generally. **Directly aligned with our PR-AUC-as-primary-metric choice.** |
| `grover2022fdb` | Grover et al. (Amazon) (2022). *Fraud Dataset Benchmark and Applications*. | `General/General_Grover_FDB_Benchmark.pdf` | Multi-dataset fraud benchmark — broader scope than ULB. |

### Tier 2 — KEEP CONDITIONAL (cite if SoA includes matching subsection)

| Citation key | Paper | File | Cite if |
|---|---|---|---|
| `almhaithawi2020smote` | Almhaithawi et al. (2020). *Example-dependent cost-sensitive credit cards fraud detection using SMOTE and Bayes minimum risk*. SN Applied Sciences. | `ULB_Almhaithawi2020_SmoteBMR.pdf` | SMOTE methodology subsection. Already curated from referencesForAproval. |
| `jurgovsky2018sequence` | Jurgovsky et al. (2018). *Sequence Classification for CCFD*. ESWA Q1. | `ULB_Jurgovsky2018_SequenceClassification.pdf` | Sequence/LSTM DL subsection. Already curated. |
| `lucas2019hmm` | Lucas et al. (2019). *Towards automated FE for CCFD using multi-perspective HMMs*. INSA Lyon + Worldline. | `ULB_Lucas2019_MultiPerspectiveHMMs.pdf` | Feature engineering historical. Already curated. |
| `baisholan2025fraudxai` | Baisholan et al. (2025). *FraudX AI: An Interpretable ML Framework for CCFD on Imbalanced Datasets*. Computers MDPI. | `ULB_Baisholan2025_FraudXAI.pdf` | XAI alternative perspective on ULB. |
| `mim2024softvoting` | Mim, Majadi, Mazumder (2024). *A soft voting ensemble learning approach for credit card fraud detection*. Heliyon (Elsevier). | `ULB_Mim2024_SoftVotingEnsemble.pdf` | Ensemble + sampling discussion. **Heliyon is legit Elsevier journal.** |
| `btoush2025excellence` | Btoush et al. (2025). *Achieving Excellence in Cyber Fraud Detection: A Hybrid ML+DL Ensemble Approach for Credit Cards*. Applied Sciences MDPI. | `ULB_Btoush2025_HybridMLDL.pdf` | Hybrid ML+DL ensemble discussion. Peer-reviewed MDPI. |

### Tier 2 — Moved to General/ (broader scope)

| Citation key | Paper | New filename | Cite if |
|---|---|---|---|
| `compagnino2025introduction` | Compagnino et al. (2025). *An Introduction to Machine Learning Methods for Fraud Detection*. Applied Sciences MDPI. | `General/General_Compagnino2025_IntroductionML.pdf` | Recent introductory ML review for fraud. Could anchor Section 1 (Fraud Detection in Financial Transactions) overview. |
| `wu2020dualgan` | Wu, Cui, Welsch (2020). *Dual Autoencoders Generative Adversarial Network for Imbalanced Classification Problem*. IEEE Access. | `General/General_Wu2020_DualAutoencodersGAN.pdf` | GAN-based oversampling — methodologically interesting if SoA discusses synthetic data generation as alternative to SMOTE. |

### Tier 3 — DROP (moved to `_dropped/`)

20 PDFs moved to `ULB/_dropped/` for these reasons:

| File | Drop reason |
|---|---|
| `ULB_A Comparative Study of Machine Learning Models for CCFD.pdf` (Chen, Yinlei 2025) | Academic Journal of Natural Science (questionable venue), sole author from non-CS dept Korea, qq.com email. |
| `ULB_A Deep Learning Method ... Continuous-Coupled Neural Networks.pdf` (Wu et al. 2025 Mathematics MDPI) | Originally `Dataset2013_*`. Exotic NN approach, not aligned with thesis framing. |
| `ULB_A Systematic Review of ML in CCFD.pdf` (Moradi 2025 preprints.org) | Preprint, not peer-reviewed. Reports RF "99.98% accuracy" — classic metric misuse. |
| `ULB_A comprehensive evaluation of oversampling techniques for enhancing text classification performance.pdf` | **Not about fraud** — text classification (TREC, Emotions datasets). Misclassified into ULB folder. |
| `ULB_A novel method for detecting credit card fraud problems.pdf` (Dataset2013 original) | Generic, low-quality. |
| `ULB_Adaptive ... RL Agents vs Anomaly Detection.pdf` (Ben Mekhlouf 2026 MDPI) | RL focus outside thesis scope. |
| `ULB_Addressing Class Imbalance ... Hybrid Deep Learning.pdf` (Khan 2025 IJIST Pakistan) | Questionable journal. AUPRC 0.886 with GRU+LSTM+SMOTE-Tomek on ULB — implausibly high (likely leakage). |
| `ULB_Addressing imbalanced data ... CRN-SMOTE.pdf` (Hemmatian, Iran) | **Doesn't use ULB** — uses ILPD/QSAR/Blood/Maternal Health. SMOTE variant not used in our thesis. |
| `ULB_An Explainable Credit Card Fraud Detection Model ... ML and DL.pdf` (Alkhozae 2025 Saudi Arabia) | Uses 1.6M transactions — **NOT ULB** (which is 284K). XGBoost 99.86% accuracy implausible. |
| `ULB_An_Explainable_Ensemble_Model_for_CCFD ...` (Karim et al. 2025 NCIM Bangladesh) | Tier-3 conference Bangladesh. |
| `ULB_Application of ML T... Credit Card Fraud.pdf` (Shakya 2018 UNLV) | Master's thesis, not peer-reviewed paper. Title corrupted by OCR. |
| `ULB_Benchmarking ML Models for Real-Time Fraud Detection in Digital Banking.pdf` (Islam 2026 posthumanism.co.uk) | Predatory/highly questionable journal. |
| `ULB_CCFD Efficient ... Meta-Heuristic Techniques.pdf` (Dataset2013 original) | Generic, low-quality. |
| `ULB_Comparative Analysis of Machine.pdf` (Wallberg & Alkattab KTH BSc) | Bachelor's degree project (15 hp), not peer-reviewed publication. |
| `ULB_Comparative Study of Data-Level Imbalance Handling.pdf` (Youssef 2025 ResearchSquare) | Preprint, generic comparison. |
| `ULB_Credit Card Fraud Detection A Realistic Modeling and a Novel Learning Strategy (ResearchGate).pdf` | **Duplicate** of `dalpozzolo2017realistic` already in `.bib`. |
| `ULB_Credit Card Fraud Detection A System Based on Imbalanced Learning ...pdf` (Li, Beijing Lang Univ) | Sole author from non-CS dept (Beijing Language and Culture University), qq.com email. |
| `ULB_Credit Card Fraud Detection Using Ensemble (Stacking and Voting) with Hybrid.pdf` | Generic title, no specific contribution identified. |
| `ULB_Credit Card Fraud Detection Using Machine Learning.pdf` (Dataset2013 original) | Generic title, low-quality. |
| `ULB_Credit Card Fraud Detection via Model Retraining and Fine-Tuning.pdf` (Khadka UTA) | UNT student research, not peer-reviewed. |

---

## Phase 2 PART 2 — Second half pending

23 PDFs not yet triaged (will require re-extraction of `.txt` for those files in next session). Subjects clustered around:
- SMOTE/imbalance (Hordri IJACSA, Hossain AJMRI, others — likely Tier 3 predatory) — **already pre-classified Tier 3 from earlier reads but not moved**
- XAI/SHAP papers (Evaluation of XAI techniques, Explainable AI XAI Analysis Using SHAP)
- Stacking/ensemble papers
- Federated Learning (Federated Learning Used to Detect CCF — was Dataset2013, likely Tier 3)
- Graph approaches (Heterogeneous Graph Auto-Encoder — likely Tier 3 graph)
- DL/GAN (Improving CCFD through Transformer-Enhanced GAN, etc.)

To be triaged in Phase 2 Part 2.

---

## Numerical summary so far

- **First half triaged**: 29 PDFs
  - Tier 1 ULB: **3** (Dal Pozzolo 2015, Leevy 2023, Singh 2025) — 2 new in this batch
  - Tier 1 General: **3** (Thimonier, Imani, Grover) — moved to General/
  - Tier 2 ULB: **6** (Almhaithawi, Jurgovsky, Lucas, Baisholan, Mim, Btoush 2025)
  - Tier 2 General: **2** (Compagnino, Wu 2020) — moved to General/
  - Tier 3: **20** dropped
  - Misclassified (still in ULB/, will move on Part 2): some SMOTE-Hossain/Hordri papers, etc.

- **`references.bib` after Part 1**: 25 → 30 entries (+5 new Tier 1).

- **Phase 2 Part 2**: 28 second-half PDFs to triage (23 with old names + 5 already-named-as-author from earlier curation).
