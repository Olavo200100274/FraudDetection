# PROJECT_STATE.md
**Last updated**: 2026-05-05  
**Status**: Experimental pipeline COMPLETE. Thesis writing in progress.

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
| ULB Credit Card Fraud 2013 | 284,807 | 0.172% (492 frauds) | PCA V1-V28 + Time + Amount | `datasets/creditcard.csv` |
| BAF Base (NeurIPS 2022) | 1,000,000 | ~1.1% (~11K frauds) | 30 features (25 num + 5 cat: payment_type, employment_status, housing_status, source, device_os) | `datasets/Base.csv` |
| BAF Variants I-V | 1M each | varies | same 30 features (III+V have x1,x2 dropped for cross-domain) | `datasets/Variant_*.csv` |

Cross-dataset transfer ULB↔BAF is not feasible (incompatible feature spaces: PCA vs semantic).

---

## Models

| Model | Key | Notes |
|-------|-----|-------|
| Logistic Regression | `logreg` | `src/models/logreg.py` |
| Random Forest | `rf` | `src/models/rf.py` |
| LightGBM | `lgbm` | `src/models/lgbm.py` |
| CatBoost | `catboost` | `src/models/catboost_model.py` |
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
- Variant II (label shift only): all models transfer well.
- Variants III, V (covariate shift + extra features): CatBoost/LGBM lose 24-31% PR-AUC.
- **Random Forest most robust**: improves on 3/5 Variants despite lower in-domain score.
- SHAP feature ranking: Jaccard=1.00 LGBM across all 6 BAF datasets (perfect stability).

### SHAP interpretability
- Top BAF features: device_os, housing_status, phone_home_valid (tree-based).
- FT-Transformer differs: name_email_similarity, income dominate (continuous features weighted more).
- Cross-model Jaccard: LGBM/CatBoost=1.00, vs FT-Transformer=0.33-0.54.

### FT-Transformer attention diagnostics
- Pattern: **Foggy Vision** (normalized entropy=0.976, nearly uniform attention over all 31 tokens).
- NOT Categorical Trap, NOT Tunnel Vision, NOT Self-Obsession.
- Confirms 0.18 PR-AUC is dataset ceiling, not model pathology.
- Script: `src/attention_analysis.py`, outputs in `results/baf_base/fttransformer/attention_analysis/`.

---

## Code structure (src/)

```
src/
├── main.py                 # Classical ML pipeline (Optuna + 5-fold CV + test eval)
├── main_transformer.py     # FT-Transformer pipeline (Optuna + holdout val + GPU)
├── threshold_study.py      # 4 threshold strategies applied post-hoc
├── cross_domain.py         # Transfer Base→Variants I-V
├── shap_analysis.py        # SHAP global/local/cross-model/cross-variant
├── attention_analysis.py   # FT-Transformer attention map extraction + visualization
├── generate_results.py     # Generate all tables/figures for thesis (→ results_thesis/)
├── data.py                 # Dataset registry + loaders
├── pdf_extract.py          # PyMuPDF utility for PDF text extraction
├── baf_baseline_summary.py # Summary plots for BAF baseline
├── save_load.py            # Artefact persistence (model, metrics, curves)
├── models/                 # logreg, rf, lgbm, catboost_model, ocsvm, fttransformer
├── strategies/balancing.py # 7 imbalance strategies
└── evaluation/metrics.py   # PR-AUC, ROC-AUC, F1, F2, Brier, bootstrap CI, threshold opts
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
