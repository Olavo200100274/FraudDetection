"""
Data loading module — supports multiple fraud detection datasets.

Each loader returns (X_train, X_test, y_train, y_test) after a
stratified 80/20 split with random_state=42.

An optional ``sample`` parameter (0 < sample <= 1) takes a
stratified subsample of the full dataset **before** splitting,
useful for smoke-testing the pipeline on large datasets.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42
# Project root — one level above src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dataset registry ──────────────────────────────────────────────────────
DATASET_REGISTRY = {}


def _register(name, csv_path, label):
    """Decorator that registers a dataset loader."""
    def decorator(fn):
        DATASET_REGISTRY[name] = {
            "loader": fn,
            "csv_path": csv_path,
            "label": label,
        }
        return fn
    return decorator


def load_dataset(name, sample=None):
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Registered dataset key (e.g. 'ulb', 'baf_base').
    sample : float or None
        If provided (0 < sample <= 1), take a stratified subsample of the
        full dataset before the train/test split.  Useful for smoke tests.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    entry = DATASET_REGISTRY[name]
    X_train, X_test, y_train, y_test = entry["loader"]()

    # ── Optional stratified subsample ─────────────────────────────────
    if sample is not None and sample < 1.0:
        import numpy as np
        # Recombine, subsample, re-split — preserves stratification
        X_all = pd.concat([X_train, X_test], axis=0)
        y_all = pd.concat([y_train, y_test], axis=0)

        n_sample = max(100, int(len(X_all) * sample))

        # Guarantee enough fraud cases for 5-fold CV (≥ 2 per fold = 10 min)
        fraud_rate = y_all.mean()
        min_fraud_needed = 10  # ≥ 2 per fold with CV=5
        min_n_for_fraud = int(np.ceil(min_fraud_needed / fraud_rate))
        n_sample = max(n_sample, min(min_n_for_fraud, len(X_all)))

        X_all, _, y_all, _ = train_test_split(
            X_all, y_all,
            train_size=n_sample,
            stratify=y_all,
            random_state=SPLIT_SEED,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=0.2,
            stratify=y_all,
            random_state=SPLIT_SEED,
        )
        print(f"  [--sample {sample}] Subsampled to {len(X_all):,} rows "
              f"({y_all.sum()} fraud, {y_all.mean():.2%} prevalence)")

    return X_train, X_test, y_train, y_test


def get_dataset_info(name):
    """Return (csv_path, label) for a registered dataset."""
    entry = DATASET_REGISTRY[name]
    return entry["csv_path"], entry["label"]


# ── ULB Credit Card 2013 ─────────────────────────────────────────────────

@_register("ulb", str(_PROJECT_ROOT / "datasets" / "creditcard_2013.csv"), "ulb_2013")
def load_ulb_data():
    """Load ULB Credit Card 2013 dataset (284 807 rows, 0.17% fraud)."""
    df = pd.read_csv(_PROJECT_ROOT / "datasets" / "creditcard_2013.csv")
    assert df.isnull().sum().sum() == 0, "Dataset contém valores ausentes."

    X = df.drop(columns="Class")
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    return X_train, X_test, y_train, y_test


# ── BAF Base (NeurIPS 2022) ──────────────────────────────────────────────

@_register("baf_base", str(_PROJECT_ROOT / "datasets" / "Base.csv"), "baf_base")
def load_baf_base_data():
    """
    Load BAF Base dataset (1 000 000 rows, ~1.1% fraud).

    Drops ``month`` — temporal metadata from the CTGAN generation process,
    not a real applicant feature.  Retaining it would let the model exploit
    an artefact of data synthesis rather than genuine fraud patterns.
    The ``source`` column is kept as a legitimate categorical feature.
    """
    df = pd.read_csv(_PROJECT_ROOT / "datasets" / "Base.csv")
    assert df.isnull().sum().sum() == 0, "Dataset contém valores ausentes."

    df = df.drop(columns=["month"])

    X = df.drop(columns="fraud_bool")
    y = df["fraud_bool"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    return X_train, X_test, y_train, y_test


# ── BAF Variants I–V (NeurIPS 2022) ─────────────────────────────────────

def _load_baf_variant(csv_name, drop_extra_cols=False):
    """
    Shared loader for BAF Variant datasets.

    Parameters
    ----------
    csv_name : str
        CSV filename under datasets/ (e.g. "Variant I.csv").
    drop_extra_cols : bool
        If True, drop columns ``x1`` and ``x2`` present in Variants III & V
        to maintain 32-column alignment with BAF Base.
    """
    df = pd.read_csv(_PROJECT_ROOT / "datasets" / csv_name)
    assert df.isnull().sum().sum() == 0, f"{csv_name}: unexpected null values."

    df = df.drop(columns=["month"])
    if drop_extra_cols:
        df = df.drop(columns=["x1", "x2"])

    X = df.drop(columns="fraud_bool")
    y = df["fraud_bool"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    return X_train, X_test, y_train, y_test


@_register("baf_var1", str(_PROJECT_ROOT / "datasets" / "Variant I.csv"), "baf_var1")
def load_baf_var1():
    """Load BAF Variant I — covariate shift."""
    return _load_baf_variant("Variant I.csv")


@_register("baf_var2", str(_PROJECT_ROOT / "datasets" / "Variant II.csv"), "baf_var2")
def load_baf_var2():
    """Load BAF Variant II — label shift."""
    return _load_baf_variant("Variant II.csv")


@_register("baf_var3", str(_PROJECT_ROOT / "datasets" / "Variant III.csv"), "baf_var3")
def load_baf_var3():
    """Load BAF Variant III — covariate shift + new features (x1, x2 dropped)."""
    return _load_baf_variant("Variant III.csv", drop_extra_cols=True)


@_register("baf_var4", str(_PROJECT_ROOT / "datasets" / "Variant IV.csv"), "baf_var4")
def load_baf_var4():
    """Load BAF Variant IV — bias conditions."""
    return _load_baf_variant("Variant IV.csv")


@_register("baf_var5", str(_PROJECT_ROOT / "datasets" / "Variant V.csv"), "baf_var5")
def load_baf_var5():
    """Load BAF Variant V — bias + new features (x1, x2 dropped)."""
    return _load_baf_variant("Variant V.csv", drop_extra_cols=True)