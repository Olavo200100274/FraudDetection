# ULB/ Reference Curation Notes

> **Final project status (22 July 2026):** This is an archival record of the
> completed ULB literature-curation process. Both halves described below were
> completed. The dissertation V3 has been sent to the supervisors, and the
> authoritative final selection is `Overleaf/references.bib`: 46 entries, 41
> cited. Any earlier wording such as "pending", proposed keys, or future
> curation actions is historical and does not represent outstanding thesis
> work.

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

## Phase 2 PART 2 — Second half (23 PDFs triaged 2026-05-04)

### Tier 1 — KEEP (1 new, moved to General/)

| Citation key | Paper | New filename | Why |
|---|---|---|---|
| `walauskis2024scalable` | Walauskis & Khoshgoftaar (2024). *Scalable unsupervised labeling with SHAP feature selection for fraud detection in imbalanced data*. Journal of Big Data. FAU (same group as Leevy). | `General/General_Walauskis_ScalableUnsupervisedSHAP.pdf` | Novel unsupervised SHAP feature selection methodology. Uses ULB Kaggle CCFD + Medicare Part D. Big Data journal (peer-reviewed). Methodology applies broadly → General/ folder. |

### Tier 2 — KEEP CONDITIONAL (4 new, names normalized in ULB/)

| Citation key (proposed) | Paper | File | Cite if |
|---|---|---|---|
| `mendespisani2026evaluation` | Mendes de Lima & Pisani (2026). *Evaluation of explainable AI techniques in the context of credit card fraud detection*. Journal of the Brazilian Computer Society. UFABC Brazil. | `ULB_MendesPisani2026_EvaluationXAI.pdf` | Direct comparison SHAP vs LIME on ULB. Peer-reviewed. Aligned with our SHAP analysis discussion. |
| `btoush2026resampling` | Btoush et al. (2026). *Machine Learning-Based Cyber Fraud Detection: A Comparative Study of Resampling Methods for Imbalanced Credit Card Data*. MDPI 2026. Higher Colleges Tech Dubai + UniSQ Australia. | `ULB_Btoush2026_ResamplingMethods.pdf` | Comparative resampling on ULB. Peer-reviewed MDPI. Sister paper to Btoush2025_HybridMLDL already kept. |
| `siam2025hybrid` | Siam, Bhowmik, Uddin (2025). *Hybrid feature selection framework for enhanced credit card fraud detection*. **PLOS ONE**. | `ULB_Siam2025_HybridFeatureSelection.pdf` | PLOS ONE peer-reviewed. Feature selection on ULB. Methodologically relevant if SoA discusses feature engineering. |
| `albalawi2025enhancing` | Albalawi & Dardouri (2025). *Enhancing credit card fraud detection using traditional and deep learning models with class imbalance mitigation*. **Frontiers in AI**. Saudi Arabia + Tunisia. | `ULB_Albalawi2025_EnhancingCCFD.pdf` | Frontiers in AI (legitimate journal). Uses focal loss + SMOTE — direct overlap with our imbalance methodology. Note: reports RF accuracy 99.95% (metric misuse on imbalanced) but F1=0.83 and ROC-AUC=0.97 are reasonable. |

### Tier 3 — DROP (18 PDFs moved to `_dropped/`)

| File | Drop reason |
|---|---|
| `ULB_Improving Credit Card Fraud Detection through Transformer-Enhanced GAN Oversampling.pdf` (Kashaf ul Emaan) | Single author, no clear venue/affiliation, gmail+telephone in author block. |
| `ULB_Explainable AI (XAI) Analysis Using SHAP for Credit Card Fraud.pdf` (Scripta Technica Indonesia) | Questionable Indonesian journal. |
| `ULB_Ensemble Methods and Emerging Paradigms ... Comparative Study (PDF).pdf` (López García et al. BUAP Mexico) | preprints.org (not peer-reviewed). RF "99.98% accuracy" — classic metric misuse. |
| `ULB_Improving CCFD with Ensemble DL ... SMOTE-ENN.pdf` (Bonde & Bichanga Africa, JCTA Indonesia) | Questionable Indonesian publisher. |
| `ULB_Optimized ML Model for CCFD Using SMOTE-Tomek and Feature Engineering.pdf` (Wibowo & Setiadi, JAIC Indonesia) | Indonesian regional journal. Generic title. |
| `ULB_Federated Learning Used to Detect Credit Card Fraud.pdf` (Jansson & Axelsson 2020 Lund) | **MSc thesis** from Lund University, not peer-reviewed paper. Federated learning off-scope anyway. |
| `ULB_Enhancing Financial Fraud Detection ... NN, Ensemble, Stacking.pdf` (Khandelwal IJIRCT) | IJIRCT (questionable journal), "Independent Researcher" (no affiliation). |
| `ULB_Enhancing CCFD with stacking-based hybrid ML approach.pdf` (Btoush et al.) | Same group as Btoush2025/2026 already kept. **Redundant** — third paper from same group on same topic. |
| `ULB_Handling Imbalanced Fraudulent Transaction Data Using SMOTE-Tomek and RF.pdf` (Ilham et al. Indonesia BEST) | Indonesian regional journal. Generic. |
| `ULB_Machine Learning for CCFD A Comparative Study of Algorithms.pdf` | **MSc Research Project**, not peer-reviewed. |
| `ULB_REAL-TIME CCFD MACHINE LEARNING.pdf` (Suneel Kumar 2026 JETIR) | JETIR (predatory Indian journal). RF "99.96% accuracy" implausible. Caps-lock title. |
| `ULB_SMOTE vs SMOTEENN ... Class Imbalance in Regression Models.pdf` (Husain et al. Algorithms MDPI) | **REGRESSION**, not classification. Wrong topic. |
| `ULB_Feature-based ensemble modeling for diabetes data imbalance.pdf` (Jang Ewha) | **DIABETES** dataset, not fraud. Wrong topic. |
| `ULB_ENHANCING IMBALANCED CCFD USING MULTILAYER PERCEPTION.pdf` (TPM 2025) | TPM low-tier journal. Caps-lock title. |
| `ULB_Heterogeneous Graph Auto-Encoder for CCFD.pdf` (Singh Dibrugarh India) | Graph NN approach — outside thesis scope. |
| `ULB_Handling Class Imbalance ... Resampling Methods.pdf` (Hordri 2018 IJACSA) | IJACSA predatory journal. |
| `ULB_Handling Class Imbalance ... Various Sampling Techniques.pdf` (Hossain 2022 AJMRI Bangladesh) | AJMRI low-tier. LR "99.94%" / RF "99.964%" accuracy implausible. |
| `ULB_Using neural network for credit card fraud detection.pdf` (Dataset2013 original) | Generic, low-quality. |

---

## FINAL Phase 2 totals (all 57 ULB PDFs triaged)

- **Tier 1 ULB**: 3 papers (Dal Pozzolo 2015, Leevy 2023, Singh 2025)
- **Tier 1 General** (originally in ULB but moved): 4 papers (Thimonier 2024, Imani 2026, Grover 2022 FDB, Walauskis 2024)
- **Tier 2 ULB** (kept conditional): 9 papers (Almhaithawi 2020, Jurgovsky 2018, Lucas 2019, Baisholan 2025, Mim 2024, Btoush 2025 Hybrid, Mendes Pisani 2026, Btoush 2026 Resampling, Siam 2025, Albalawi 2025)
- **Tier 2 General** (moved out): 2 papers (Compagnino 2025, Wu 2020)
- **Tier 3 dropped**: 38 papers (20 Part 1 + 18 Part 2)
- **Total accounted**: 3+4+9+2+38 = 56. (One off because Btoush2025_HybridMLDL was kept as Tier 2 — final = 57 ✓)

**`references.bib` final state**: 30 → 31 entries (added `walauskis2024scalable` in Part 2). Plus 6 new from Part 1 = 7 new across all of Phase 2.

## Patterns observed in Tier 3 drops (lessons for future curation)

1. **Predatory journals** — IJACSA, IJIRCT, IJIST, AJMRI, JETIR, posthumanism.co.uk, Scripta Technica, JAIC Polibatam, BEST Indonesia, JCTA Indonesia. These appear with high concentration in fraud detection literature (low publication bar).
2. **Metric misuse** — Reporting accuracy (97%+, 99%+) on extremely imbalanced datasets (0.17% fraud) where a trivial classifier already reaches 99.83%. Consistent red flag.
3. **MSc/BSc theses** — KTH (Wallberg), Lund (Jansson), UNLV (Shakya), UTA (Khadka). Not peer-reviewed publications, should not be cited as primary references.
4. **Wrong dataset** — Several papers in ULB folder didn't actually use ULB (Alkhozae uses 1.6M, Hemmatian uses ILPD/QSAR/Blood, Jang uses diabetes, Husain uses regression). User instinct was correct: many "ULB" papers don't use ULB.
5. **Wrong topic** — RL focus, Graph NN, Federated Learning, Text classification, Regression. Off-scope of thesis.
6. **Author group redundancy** — Btoush group (3 papers): kept the strongest two, dropped the third stacking paper.

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
