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