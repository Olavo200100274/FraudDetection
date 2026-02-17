from lightgbm import LGBMClassifier

def build_model(seed: int):
    return LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
        verbosity=-1,
    )
