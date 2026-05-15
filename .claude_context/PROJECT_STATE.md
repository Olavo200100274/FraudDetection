# PROJECT_STATE.md
**Last updated**: 2026-05-14
**Status**: Experimental pipeline ✅ COMPLETE. Thesis ✅ COMPLETE. Article 1 ✅ first draft (awaiting Maryam review). Article 2 ✅ reframed + deep-reviewed draft (awaiting Maryam review).

---

## What this project is

MSc dissertation in Applied Informatics (Universidade de Coimbra / ISCTE).  
Topic: **Fraud detection in financial transactions under extreme class imbalance.**  
Student: Olavo Caixeiro  
Advisor: Maryam Abbasi  

---

## Datasets

| Dataset | N | Fraud rate | Features | Location |
|---------|---|-----------|----------|----------|
| ULB Credit Card Fraud 2013 | 284,807 | 0.172% (492 frauds) | PCA V1-V28 + Time + Amount | `datasets/creditcard_2013.csv` |
| BAF Base (NeurIPS 2022) | 1,000,000 | ~1.1% (~11K frauds) | 30 features (25 num + 5 cat: payment_type, employment_status, housing_status, source, device_os) | `datasets/Base.csv` |
| BAF Variants I-V | 1M each | varies | same 30 features (III+V have x1,x2 dropped for cross-domain) | `datasets/Variant I.csv` ... `Variant V.csv` |

**Note on data tracking**: All `datasets/*.csv` are tracked via Git LFS (see `.gitattributes`). Same for `*.joblib`, `*.pt`, `*.npy`.

Cross-dataset transfer ULB↔BAF is not feasible (incompatible feature spaces: PCA vs semantic).

---

## Models

| Model | Key | Notes |
|-------|-----|-------|
| Logistic Regression | `logreg` | `src/models/logreg.py` |
| Random Forest | `rf` | `src/models/rf.py` |
| LightGBM | `lgbm` | `src/models/lgbm.py` |
| CatBoost | `catboost` | `src/models/catboost.py` |
| One-Class SVM | `ocsvm` | `src/models/ocsvm.py` — anomaly detection baseline only |
| FT-Transformer | `fttransformer` | `src/models/fttransformer.py` — PyTorch, GPU, holdout HPO |

---

## Imbalance strategies (7)

`none`, `rus`, `ros`, `smote`, `smote_tomek`, `smoteenn`, `weights`  
Defined in `src/strategies/balancing.py`.  
OCSVM excluded from strategy factorial.

---

## Evaluation protocol (leakage-free)

- **Outer split**: stratified 80/20 holdout (SPLIT_SEED=42). Test set NEVER touched during training/tuning/threshold.
- **Inner CV**: 5-fold stratified for classical ML (within DEV set).
- **HPO**: Optuna TPE, 50 trials, PR-AUC objective. Median threshold across folds applied to test.
- **FT-Transformer**: 80/20 within DEV (single holdout val) — GPU cost prohibits full CV.
- **Threshold strategies**: 4 tested (Fixed 0.5, max-F1, max-F2, Prec≥0.5). Primary: max-F2.
- **Primary metric**: PR-AUC (average_precision_score). Secondary: ROC-AUC, F1, F2, Brier.

---

## Key experimental results (validated in Results.tex + raw JSON)

### Baseline (strategy=None) — PR-AUC

| Model | ULB | BAF |
|-------|-----|-----|
| LR | 0.742 | 0.143 |
| RF | 0.867 | 0.159 |
| LGBM | 0.874 | 0.177 |
| **CatBoost** | **0.885** | 0.180 |
| **FT-Transformer** | 0.856 | **0.180** (ties CatBoost!) |
| OCSVM | 0.335 | 0.019 |

**Critical insight**: FT-Transformer matches CatBoost on BAF. On ULB, LGBM/CatBoost lead by ~3%. The 0.18 PR-AUC on BAF is the DATASET CEILING (16× random baseline) — confirmed by literature.

### Threshold study
- Fixed τ=0.5 is **catastrophic** on BAF: F2 near zero for all models.
- max-F2 recovers F2 ~0.30+ across supervised models.
- On ULB: τ=0.5 works reasonably for tree-based models (well-calibrated).

### Factorial (7×5, main finding)
- RUS degrades all models on both datasets.
- SMOTE degrades FT-Transformer by 41% on BAF (0.180→0.106). CatBoost: only 14%.
- SMOTE+Tomek = SMOTE (Tomek removes zero examples under extreme imbalance).
- Class Weights = safest strategy for FT-Transformer.
- Gradient boosting already handles moderate imbalance (BAF) without resampling.

### Cross-domain (BAF Base → Variants I-V)
- Variant II (label-shift dominant): all supervised models transfer well (within 0.01–0.02 of Base PR-AUC).
- Variants III and V (the hardest variants): CatBoost/LGBM/FT-T lose 24–34% PR-AUC.
- **Random Forest most robust**: improves on 4 of 5 BAF Variants (PR-AUC: I=0.206, II=0.225, III=0.170, IV=0.211, V=0.159) despite trailing the GBDTs in-domain (RF Base=0.159 vs LGBM/CatBoost 0.177/0.180). RF's F₂ remains above Base on Variants I (0.307) and IV (0.314), ties Base on III (0.289 vs 0.294, within noise), drops on V (0.278). RF is, jointly with CatBoost-under-SMOTE, one of two places where a tree-based model strictly beats FT-T.
- SHAP feature ranking: Jaccard=1.00 LGBM across all 6 BAF datasets (perfect stability — explanations remain valid under shift without recomputation).

### SHAP interpretability
- Top BAF features: device_os, housing_status, phone_home_valid (tree-based).
- FT-Transformer differs: name_email_similarity, income dominate (continuous features weighted more).
- Cross-model Jaccard: LGBM/CatBoost=1.00, vs FT-Transformer=0.33-0.54.

### FT-Transformer attention diagnostics
- Pattern: **Foggy Vision** (normalized entropy=0.976, nearly uniform attention over all 31 tokens).
- NOT Categorical Trap (cat/num ratio = 1.28×), NOT Tunnel Vision (top-3 share <16%), NOT Self-Obsession ([CLS] self-attn = 3.1% ≈ 1/31).
- Top-3 attention tokens: `device_os` (6.2%), `housing_status` (5.0%), `employment_status` (3.8%).
- Confirms 0.18 PR-AUC is dataset ceiling, not model pathology.
- Script: `src/attention_analysis.py`, outputs in `results/baf_base/fttransformer/attention_analysis/`.

### Convergent validity (Article 2 contribution)
- LGBM SHAP top-10 vs FT-T attention top-10 on the same BAF Base test set.
- **Top-2 identical in both rankings**: `device_os` (rank 1), `housing_status` (rank 2).
- **5 of 10 features overlap** between the two rankings: device_os, housing_status, phone_home_valid, income, current_address_months_count.
- Two architecturally different models (gradient-boosted forest vs self-attention stack) + two methodologically different attribution estimators (Shapley values vs learned attention weights) → unusually robust evidence.
- Article 2 §5.5 Pillar 4: bounds the architectural-improvement headroom above the 0.18 PR-AUC ceiling.

---

## Code structure (src/)

```
src/
├── main.py                              # Classical ML pipeline (Optuna + 5-fold CV + test eval)
├── main_transformer.py                  # FT-Transformer pipeline (Optuna + holdout val + GPU)
├── run_all_transformer.py               # Sequential runner: threshold ULB+BAF + cross-domain for FT-T
├── threshold_study.py                   # 4 threshold strategies applied post-hoc
├── cross_domain.py                      # Transfer Base→Variants I-V
├── shap_analysis.py                     # SHAP global/local/cross-model/cross-variant
├── attention_analysis.py                # FT-Transformer attention map extraction + visualization
├── generate_results.py                  # Generate all tables/figures for thesis (→ results_thesis/)
├── baf_baseline_summary.py              # Summary plots for BAF baseline
├── generate_pr_curves_threshold.py      # ARTICLE 1: PR curves with threshold-rule markers (ULB+BAF) — added 2026-05-11
├── generate_crossdomain_chart.py        # ARTICLE 2: cross-domain PR-AUC line chart (RF advantage) — added 2026-05-13
├── generate_smote_asymmetry_chart.py    # ARTICLE 2: FT-T vs CatBoost across 7 strategies × 2 datasets — added 2026-05-14
├── generate_convergent_validity_chart.py # ARTICLE 2: SHAP top-10 vs Attention top-10 with overlap shading — added 2026-05-14
├── data.py                              # Dataset registry + loaders
├── preprocess.py                        # ColumnTransformer (impute + scale numeric, one-hot categorical)
├── pdf_extract.py                       # PyMuPDF utility for PDF text extraction (used during ref curation)
├── save_load.py                         # Artefact persistence (model, metrics, curves)
├── models/                              # logreg.py, rf.py, lgbm.py, catboost.py, ocsvm.py, fttransformer.py
├── strategies/balancing.py              # 7 imbalance strategies + sampler factory
└── evaluation/metrics.py                # PR-AUC, ROC-AUC, F1, F2, Brier, bootstrap CI, threshold opts
```

## Notebooks

```
notebooks/
├── eda_ulb.ipynb           # Exploratory data analysis on ULB 2013
├── eda_baf.ipynb           # EDA on BAF Base
└── baf_variants_eda.ipynb  # EDA across BAF Variants I-V
```

## Output / artefact directories

```
results/                  # Active experimental output (committed via Git LFS)
results_thesis/           # Generated thesis-ready figures + tables (regenerated by generate_results.py)
├── figures/baf/   # 9 PDFs (PR curves, heatmaps, factorial bars, etc.)
├── figures/ulb/   # 5 PDFs
└── tables/        # legacy structure (most tables now inline in 4-Results.tex)
results_old_gridSearch/   # ARCHIVED: pre-Optuna GridSearch results (gitignored, do not use)
catboost_info/            # CatBoost training logs (gitignored)
```

---

## Known code issue (bug, not fixed yet)

**Logistic Regression `l1_ratio` silently ignored**: `LogisticRegression(solver='saga')` — Optuna suggests `l1_ratio` but only applies with `penalty='elasticnet'`. Current code uses default `penalty='l2'`. Fix: add `penalty='elasticnet'` to LR. LR is worst model so impact on thesis conclusions is minimal.

---

## Hardware/reproducibility
- CPU: Intel i5-13600KF (14 cores), 16GB RAM  
- GPU: NVIDIA RTX 3070 (8GB VRAM) — used for FT-Transformer  
- All seeds: 42  
- SHA-256 hash logged per dataset  
- Full config JSON per run  

---

## Results folder structure

```
results/
├── baf_base/
│   ├── catboost/none/run_*/   {config.json, model.joblib, metrics_*.json, shap_*, threshold_study.json, cross_domain.json}
│   ├── lgbm/, rf/, logreg/, ocsvm/
│   └── fttransformer/none/run_*/  {model.pt instead of .joblib} + attention_analysis/
└── ulb_2013/
    └── [same structure, no cross_domain.json]
```
