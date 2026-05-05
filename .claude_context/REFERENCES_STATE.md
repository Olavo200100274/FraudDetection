# REFERENCES_STATE.md
**Last updated**: 2026-05-05  
**Total .bib entries**: 43  

---

## Folder structure

```
Overleaf/references/
├── BAF/         10 PDFs (Tier 1+2) + 9 in _dropped/
├── ULB/         13 PDFs (Tier 1+2) + 38 in _dropped/
├── General/     27 PDFs (methodological, surveys, foundational)
└── referencesForAproval/  EMPTY (all processed)
```

---

## Current references.bib — all 41 keys

### ULB-foundational
- `dalpozzolo2017realistic` — Pozzolo 2017, IEEE TNNLS (~700 cit) — ULB modeling
- `dalpozzolo2015calibrating` — Pozzolo 2015, IEEE SSCI (~600 cit) — calibration + undersampling on ULB
- `carcillo2017scarff` — Carcillo 2017, Inf. Fusion (~150 cit) — streaming detection on ULB
- `carcillo2018streamingAL` — Carcillo 2018, Inf. Sciences (~250 cit) — active learning on ULB
- `hayat2025leakage` — Hayat 2025 — **CRITICAL**: critiques data leakage in fraud detection literature

### BAF-foundational
- `jesus2022turning` — Jesus et al. 2022, NeurIPS Datasets (~200 cit) — BAF dataset paper
- `cruz2023fairgbm` — Cruz et al. 2023, ICLR (~50 cit) — FairGBM on BAF/AOF, fairness GBDT
- `sun2025objective` — Sun et al. 2025, Computation MDPI — objective > architecture on BAF Base
- `luzio2024decoupling` — Luzio et al. 2024, ACM-SAC — calibration/threshold decoupling on BAF
- `alves2025openl2d` — Alves et al. 2025, Nature Sci Data — OpenL2D/FiFAR framework on BAF

### Tabular DL (models used in thesis)
- `gorishniy2021revisiting` — Gorishniy 2021, NeurIPS (~700 cit) — **FT-Transformer original paper**
- `huang2020tabtransformer` — Huang et al. 2020, arXiv (~700 cit) — TabTransformer (categorical attention)
- `somepalli2021saint` — Somepalli et al. 2021, arXiv (~250 cit) — SAINT (row+col attention, claims to beat GBDT)
- `gorishniy2022embeddings` — Gorishniy 2022, NeurIPS (~150 cit) — Periodic embeddings for numerical features

### Tabular DL surveys (the "DL vs GBDT" debate)
- `shwartzziv2022tabular` — Shwartz-Ziv & Armon 2022, Inf. Fusion (~650 cit) — **"DL is not all you need"** — CENTRAL to Maryam framing
- `borisov2022deep` — Borisov et al. 2022, IEEE TNNLS (~850 cit) — comprehensive tabular DL survey

### Class imbalance (foundationals)
- `chawla2002smote` — Chawla 2002, JAIR (**35,000+ cit**) — SMOTE original
- `batista2004balancing` — Batista 2004, ACM SIGKDD (**3,500+ cit**) — SMOTE+Tomek and SMOTEENN original
- `hegarcia2009learning` — He & Garcia 2009, IEEE TKDE (**11,000+ cit**) — classical imbalance survey
- `krawczyk2016learning` — Krawczyk 2016, Prog AI (**2,000+ cit**) — modern imbalance survey

### Threshold + evaluation
- `saito2015precision` — Saito & Rehmsmeier 2015, PLOS ONE (**3,000+ cit**) — **PR-AUC > ROC-AUC under imbalance** — justifies our primary metric
- `imani2026why` — Imani et al. 2026, MDPI — Why ROC-AUC misleading, uses ULB, empirically confirms PR-AUC advantage
- `nesvijevskaia2021accuracy` — Nesvijevskaia 2021, Data & Policy — accuracy vs interpretability in fraud detection
- `leevy2023binary` — Leevy 2023, J. Big Data — binary vs one-class classification on ULB, CatBoost wins

### XAI (explainability)
- `lundberg2017unified` — Lundberg & Lee 2017, NeurIPS (**20,000+ cit**) — **SHAP original paper**
- `lundberg2020local` — Lundberg et al. 2020, Nature MI (**3,500+ cit**) — Tree SHAP exact
- `ribeiro2016lime` — Ribeiro et al. 2016, KDD (**17,000+ cit**) — LIME paper (XAI comparison)
- `molnar2022interpretable` — Molnar 2022, Book (**2,000+ cit**) — Interpretable ML book
- `walauskis2024scalable` — Walauskis & Khoshgoftaar 2024, J. Big Data — unsupervised SHAP feature selection

### Transformers (foundational)
- `vaswani2017attention` — Vaswani et al. 2017, NeurIPS (**130,000+ cit**) — **Attention Is All You Need**

### Surveys (fraud detection)
- `rymantubb2018survey` — Ryman-Tubb 2018, Eng. Appl. AI (~500 cit) — comprehensive fraud+ML survey
- `hilal2022financial` — Hilal et al. 2022, ESWA (~250 cit) — anomaly detection in financial fraud survey
- `ali2022financial` — Ali et al. 2022, Appl. Sci. (~150 cit) — systematic literature review (93 articles)
- `hernandezaros2024financial` — Hernandez Aros 2024, HSSC (recent) — 2024 literature review
- `mienye2024deep` — Mienye & Jere 2024, IEEE Access (recent) — DL for credit card fraud review

### Comparative studies on fraud
- `singh2025benchmarking` — Singh 2025, Comp. Econ. Springer — 15 ML algorithms + TabNet on ULB + SHAP
- `thimonier2024comparative` — Thimonier 2024, Paris-Saclay — **"LightGBM significantly outperforms anomaly detection"** on credit card fraud (directly supports Maryam framing)
- `grover2022fdb` — Grover Amazon 2022, arXiv — Fraud Dataset Benchmark (multi-dataset API)

### Methodology tools
- `akiba2019optuna` — Akiba 2019, KDD (**3,000+ cit**) — Optuna TPE sampler
- `hoppner2021idcs` — Höppner 2021, EJOR — instance-dependent cost-sensitive credit card fraud
- `yesilkanat2020adaptive` — Yesilkanat 2020, App. Soft Comp — adaptive credit card fraud detection

### GBDT implementations (added 2026-05-05 for SoA)
- `ke2017lightgbm` — Ke et al. 2017, NeurIPS — LightGBM original paper (leaf-wise growth, histogram-based gradients)
- `prokhorenkova2018catboost` — Prokhorenkova et al. 2018, NeurIPS — CatBoost original paper (ordered boosting, native categoricals)

---

## Coverage gaps: NONE remaining

All 6 thesis contributions now have supporting literature:
1. Unified comparison: shwartzziv2022, borisov2022, hayat2025, thimonier2024
2. Threshold: saito2015, imani2026, luzio2024, nesvijevskaia2021
3. Factorial (imbalance): chawla2002, batista2004, hegarcia2009, krawczyk2016
4. Cross-domain: jesus2022 (Variants designed for this)
5. SHAP: lundberg2017+2020, ribeiro2016, molnar2022
6. FT-T attention: vaswani2017, gorishniy2021+2022

---

## Papers NOT in .bib but in General/ folder (Tier 2 — conditional)

These are in the folder for potential use but NOT yet in .bib.  
Add to .bib at SoA writing time if the relevant subsection materialises:

- `General_Compagnino2025_IntroductionML.pdf` — recent intro review Applied Sci
- `General_Wu2020_DualAutoencodersGAN.pdf` — GAN oversampling for imbalance (IEEE Access)
- `General_Shenkar2022_AnomalyDetectionTabular.pdf` — anomaly detection on tabular (ICLR 2022)
- `General_Cartella2021_AdversarialTabular.pdf` — adversarial attacks on tabular fraud (arXiv)
- `General_Ti2022_FeatureGeneration.pdf` — feature generation comparison (Nature SR)
- `General_Amarasinghe2018_CriticalAnalysis.pdf` — critical ML analysis fraud (ACM)
- `ULB_Baisholan2025_FraudXAI.pdf` — FraudX AI interpretable framework (Computers MDPI)
- `ULB_Mim2024_SoftVotingEnsemble.pdf` — soft voting ensemble (Heliyon Elsevier)
- `ULB_Btoush2025_HybridMLDL.pdf` — hybrid ML+DL on ULB (Applied Sciences MDPI)
- `ULB_MendesPisani2026_EvaluationXAI.pdf` — SHAP vs LIME evaluation (JBCS Brazil)
- `ULB_Btoush2026_ResamplingMethods.pdf` — resampling comparison (MDPI)
- `ULB_Siam2025_HybridFeatureSelection.pdf` — feature selection (PLOS ONE)
- `ULB_Albalawi2025_EnhancingCCFD.pdf` — focal loss + SMOTE (Frontiers AI)
- `BAF_Neuromorphic...` — SNN on BAF (Coimbra)
- `BAF_Understanding Unfairness...` — fairness on BAF (KDD Workshop)
- `BAF_Reinforcement-Guided...` — SNN+RL on BAF (Knowledge-Based Systems)
