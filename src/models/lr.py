from sklearn.linear_model import LogisticRegression

def build_model(seed: int):
    # class_weight balanced é um baseline sólido para D1
    return LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
