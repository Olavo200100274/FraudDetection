from sklearn.ensemble import RandomForestClassifier

def build_model(seed: int):
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
