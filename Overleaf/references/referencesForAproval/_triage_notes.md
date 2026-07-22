# referencesForAproval/ Triage Notes

> **Final project status (22 July 2026):** This queue is closed and this file is
> retained only as an audit trail. The dissertation is complete and the V3 PDF
> has been sent to the supervisors. The authoritative bibliography is
> `Overleaf/references.bib`, with 46 entries and 41 cited keys. No triage or
> conditional bibliography action remains pending for submission.

**Triaged by**: Claude Opus 4.7 (1M context)
**Date**: 2026-05-04
**Total files triaged**: 46 (42 PDFs + 4 .url)

This folder was a triage queue. After Phase 3 of the SoA curation plan, all files have been redistributed:

---

## Outcome summary

- **Tier 1 (8 PDFs + 1 .url) → moved to `General/` or `ULB/`** — added to `references.bib`
- **Tier 2 (7 PDFs) → moved to `General/` or `ULB/`** — `.bib` entries pending until SoA writing confirms section
- **Tier 3 (27 PDFs) → moved to `_dropped/`** — preserved for audit, not cited
- **Links (3 .url) → moved to `_links/` subfolders** — referenced via `\url{}`, no `.bib` entry

After Phase 3 the `referencesForAproval/` folder is empty (only `_dropped/` remains).

---

## Mapping: file → destination + tier

### Tier 1 → KEEP

| Original filename | Destination | New filename | BibTeX key |
|---|---|---|---|
| `11 TabTransformer Tabular Data Modeling.pdf` | `General/` | `General_Huang2020_TabTransformer.pdf` | `huang2020tabtransformer` |
| `13 SAINT Improved Neural Networks for Tabular Data.pdf` | `General/` | `General_Somepalli2021_SAINT.pdf` | `somepalli2021saint` |
| `24 Calibrating Probability with Undersampling.pdf` | `ULB/` | `ULB_DalPozzolo2015_CalibratingUndersampling.pdf` | `dalpozzolo2015calibrating` |
| `8 Financial Fraud A Review of Anomaly Detection...` | `General/` | `General_Hilal2022_AnomalyDetectionReview.pdf` | `hilal2022financial` |
| `11 Financial fraud detection through... literature review.pdf` | `General/` | `General_HernandezAros2024_MLReview.pdf` | `hernandezaros2024financial` |
| `9 Financial Fraud Detection Based on ML A SLR.pdf` | `General/` | `General_Ali2022_MLSLR.pdf` | `ali2022financial` |
| `15 The accuracy versus interpretability trade-off in fraud.pdf` | `General/` | `General_Nesvijevskaia2021_AccuracyVsInterpretability.pdf` | `nesvijevskaia2021accuracy` |
| `28 Deep Learning for Credit Card Fraud Detection A Review of Algorithms...` | `General/` | `General_Mienye2024_DLReview.pdf` | `mienye2024deep` |
| `18 Molnar, C. (2022). _Interpretable machine learning.url` | `General/` | `General_Molnar2022_InterpretableML.url` | `molnar2022interpretable` |

### Tier 2 → KEEP CONDITIONAL

| Original filename | Destination | New filename | Cite if |
|---|---|---|---|
| `14 ANOMALY DETECTION FOR TABULAR DATA WITH.pdf` (Shenkar & Wolf 2022 ICLR) | `General/` | `General_Shenkar2022_AnomalyDetectionTabular.pdf` | OCSVM/anomaly subsection |
| `18 Adversarial Attacks for Tabular Data...` (Cartella 2021 Sony) | `General/` | `General_Cartella2021_AdversarialTabular.pdf` | Adversarial subsection |
| `5 Sequence classification for credit-card fraud detection.pdf` (Jurgovsky 2018) | `ULB/` | `ULB_Jurgovsky2018_SequenceClassification.pdf` | Sequence/LSTM DL subsection |
| `6 Towards automated FE for CCFD using HMMs.pdf` (Lucas 2019) | `ULB/` | `ULB_Lucas2019_MultiPerspectiveHMMs.pdf` | Feature engineering subsection |
| `12 Feature generation and contribution comparison.pdf` (Ti 2022 Nature SR) | `General/` | `General_Ti2022_FeatureGeneration.pdf` | SHAP feature importance subsection |
| `6 Critical Analysis of ML Based Approaches.pdf` (Amarasinghe 2018) | `General/` | `General_Amarasinghe2018_CriticalAnalysis.pdf` | Methodological critique subsection |
| `7 Example-dependent cost-sensitive credit cards fraud detection using.pdf` (Almhaithawi 2020 SN App Sci) | `ULB/` | `ULB_Almhaithawi2020_SmoteBMR.pdf` | SMOTE + cost-sensitive comparison subsection |

### Tier 3 → DROP (in `_dropped/`)

#### Already in `references.bib` under different name (10 files):

| Original filename | Already in `.bib` as |
|---|---|
| `1 How AI and ML research impacts payment card fraud detection A survey...` | `rymantubb2018survey` |
| `2 Credit Card Fraud Detection A Realistic Modeling and a Novel Learning Strategy` | `dalpozzolo2017realistic` |
| `3 SCARFF...Spark` | `carcillo2017scarff` |
| `4 Streaming Active Learning Strategies...` | `carcillo2018streamingAL` |
| `5 Instance-dependent cost-sensitive learning...` | `hoppner2021idcs` |
| `6 An Adaptive Approach...Word Embeddings` | `yesilkanat2020adaptive` |
| `7 Turning the Tables...` | `jesus2022turning` |
| `8 Data Leakage and Deceptive Performance...` | `hayat2025leakage` |
| `12 Revisiting Deep Learning Models for Tabular Data` | `gorishniy2021revisiting` |
| `16 Turning the Tables Biased, Imbalanced, Dynamic` | `jesus2022turning` (duplicado) |

#### Internal duplicate (1 file):

| Filename | Duplicates |
|---|---|
| `25 Deep Learning for Credit Card Fraud Detection.pdf` | `28 Deep Learning for CCFD A Review...` (mesmo paper Mienye & Jere 2024, ficheiros idênticos 6,219,527 bytes) |

#### Off-scope or low-quality (16 files):

| Filename | Drop reason |
|---|---|
| `00__Referencias_Bibliograficas.pdf` | Lista de referências, não é paper. |
| `10 An intelligent payment card fraud detection system.pdf` | Genérico, sem mérito específico. |
| `15 Dynamic Graph Neural Networks for Multi-Level Financial Fraud Detection...` | Graph NN — fora do scope (tabular). |
| `16 Credit Card Fraud Detection with XAI Improving.pdf` (Vihurskyi 2024 IEEE ICDCECE) | Reporta 100% accuracy DT/RF — implausível, conf. baixo tier. |
| `17 Graph Neural Network for Fraud Detection via Spatial-Temporal Attention.pdf` | Graph NN — fora do scope. |
| `20 Fraud Detection Using the Fraud Triangle Theory and Data.pdf` | Fraud Triangle = framework auditoria contabilística, fora ML scope. |
| `21 Key Considerations to be Applied While.pdf` | Título vago, conteúdo genérico. |
| `22 Comparative Review of Credit Card.pdf` | Review genérico baixo tier. |
| `23 Intelligent Fraud Detection in Financial.pdf` | Título genérico, paper genérico. |
| `26 Online Payment Fraud Detection Model Using.pdf` | Genérico. |
| `27 Real-Time Fraud Detection Using Machine.pdf` | Genérico. |
| `3 FRAUD DETECTION IN FINANCIAL TRANSACTIONS.pdf` | Título genérico (caps lock), paper genérico. |
| `5 Machine_Learning-Based_Real-Time_Fraud_Detection.pdf` | Genérico. |
| `A Data-Driven Framework For Real-Time Fraud Detection... Big Data Analytics.pdf` | Genérico, big data buzzword. |
| `Review_on_fraud_detection_methods_in_credit_card_transactions.pdf` (Modi & Dayma) | 5-page review, baixo tier (LD College Engineering India). |
| `Transparency_and_Privacy_The_Role_of_Explainable_AI_and_Federated_Learning.pdf` (Awosika 2024) | Combina FL+XAI; FL fora do scope. |

### Links (3 .url) → moved to `_links/` subfolders

| Original | Destination |
|---|---|
| `5 Credit Card Fraud Detection ULB 2013.url` | `ULB/_links/` |
| `6 Credit Card Fraud Detection Dataset 2023.url` | `_links/` (root, not used in thesis but preserved) |
| `7 Bank Account Fraud Dataset Suite (NeurIPS 2022).url` | `BAF/_links/` |

---

## Failed extraction recovered

`7 Example‑dependent cost‑sensitive credit cards fraud detection using.pdf` initially failed extraction due to Unicode hyphen (U+2011) in the filename. After renaming to ASCII hyphen, the paper was identified as Almhaithawi et al. (2020) "Example-dependent cost-sensitive credit cards fraud detection using SMOTE and Bayes minimum risk", SN Applied Sciences (Springer). Classified as **Tier 2 ULB** and moved to `ULB/ULB_Almhaithawi2020_SmoteBMR.pdf`.
