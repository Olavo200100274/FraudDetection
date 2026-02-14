from catboost import CatBoostClassifier

def build_model(seed: int):
    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        auto_class_weights="Balanced",
    )
