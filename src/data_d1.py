import pandas as pd

def load_d1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic sanity checks
    expected = {"Time", "Amount", "Class"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"D1 missing columns: {missing}")

    # Remove exact duplicates to reduce evaluation inflation risk
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[D1] drop_duplicates: {before} -> {after} (removed {before-after})")

    # Ensure target is int
    df["Class"] = df["Class"].astype(int)

    return df
