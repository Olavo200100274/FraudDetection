# Financial Fraud Detection under Extreme Class Imbalance

This repository contains the source code, saved experimental artefacts,
dissertation source, article manuscripts, and final submission candidate for
Olavo Caixeiro's MSc dissertation in Applied Informatics.

## Current status

**Status date:** 22 July 2026

**Completion state:** Phases 1--6 completed

**Submission state:** The final V3 dissertation was sent to the supervisors on
22 July 2026. No scientific, editorial, structural, or visual correction is
currently outstanding. The project is waiting only for possible final feedback
and the formal institutional deposit.

The accepted submission candidate is:

- `deliverables/2026_22_07_Thesis_MSc_Olavo_V3.pdf`

The supervisor review and the accompanying change summary are preserved in the
same directory. See `deliverables/README.md` for the exact file roles.

## Scientific scope

The dissertation studies financial fraud detection under extreme class
imbalance through five research questions:

1. Comparative model performance under a leakage-free protocol.
2. The effect of validation-derived decision-threshold selection.
3. Interactions between imbalance strategy and model family.
4. Generalisation from BAF Base to the five controlled BAF variants.
5. SHAP feature-set consistency and FT-Transformer attention diagnostics.

The experimental design uses:

- ULB Credit Card Fraud 2013 and the Bank Account Fraud suite;
- Logistic Regression, Random Forest, LightGBM, CatBoost, FT-Transformer, and
  One-Class SVM;
- seven imbalance configurations for the five supervised models;
- four threshold rules selected without TEST access;
- a fixed BAF Base-to-Variant transfer protocol;
- SHAP analyses for compatible explainers and a separate attention diagnostic.

No additional experimental run is required to support the submitted thesis.
Future work must not replace the accepted saved values unless a new scope is
explicitly authorised and documented.

## Repository structure

| Path | Role |
|---|---|
| `Overleaf/` | Complete LaTeX dissertation source. `Overleaf/main.tex` is the entry point. |
| `src/` | Experimental, evaluation, interpretability, and result-generation code. |
| `datasets/` | ULB and BAF datasets tracked through Git LFS. |
| `results/` | Saved models, scores, metrics, and analysis artefacts. Large binary files use Git LFS. |
| `results_thesis/` | Generated tables and figures used by the dissertation and articles. |
| `notebooks/` | Exploratory analysis notebooks for ULB and BAF. |
| `Article 1/` | First article manuscript and Elsevier template resources. |
| `Article 2/` | Second article manuscript, figures, and the CAS files required for local compilation. |
| `deliverables/` | Final dissertation PDF, supervisor review, and change-summary document. |

The local `.claude_context/` directory is intentionally excluded from version
control because it contains assistant-specific working context. This README and
the deliverables manifest preserve the durable project state required after a
computer reset without publishing that private working material.

## Final dissertation validation

The final V3 PDF passed a complete cover-to-cover review of all 111 physical
pages. The final checks confirmed:

- coherent alignment between the five research questions, six contributions,
  methodology, results, and conclusion;
- agreement between the narrative claims and the saved/displayed values;
- 124 unique LaTeX labels and 127 reference uses with no undefined reference;
- 24 `\pageref` commands and 41 cited bibliography entries;
- 17 labelled figures and 33 labelled tables;
- consistent British English in the scientific text;
- Portuguese Resumo and personal Agradecimentos as authorised front matter;
- no unresolved placeholder, broken citation, clipped element, or trailing
  blank page;
- only the expected `openright` and front-matter blank pages.

The Abstract and Resumo each contain exactly 150 words. The current PDF contains
Roman page labels `i`--`xx`, followed by Arabic page labels `1`--`91`.

## Restoring the project on a new computer

1. Install Git and Git LFS.
2. Clone the repository and fetch all LFS objects:

   ```bash
   git clone https://github.com/Olavo200100274/FraudDetection.git
   cd FraudDetection
   git lfs install
   git lfs pull
   ```

3. Create a Python virtual environment and install the recorded project
   dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   The recorded environment uses the CUDA 12.1 PyTorch build. On a machine
   without a compatible NVIDIA setup, install an appropriate CPU or current
   CUDA build of PyTorch before running transformer code.

4. Import the contents of `Overleaf/` into Overleaf and compile
   `Overleaf/main.tex` with pdfTeX. The versioned PDF in `deliverables/` remains
   the authoritative submitted V3 candidate.

## Writing and language policy

- Dissertation chapters, captions, tables, article text, code, comments, and
  repository documentation are written in English.
- The scientific writing standard is British English, using forms such as
  `generalisation`, `optimisation`, `modelling`, and `behaviour`.
- The authorised Portuguese thesis content consists of the Resumo and the
  author's personal Agradecimentos.

## Next action

Wait for possible supervisor or institutional feedback. If feedback arrives,
record it before changing the accepted source, make only evidence-supported
changes, rebuild the PDF in Overleaf, and re-run the relevant textual, numeric,
reference, pagination, and visual checks before formal deposit.
