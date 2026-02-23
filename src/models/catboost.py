from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline


def get_pipeline_and_params(preprocessor):
    """Return (pipeline, param_grid) for CatBoost.

    Grid rationale (ULB 2013 / tabular fraud):
    - iterations 200-800 with learning_rate 0.01-0.1 covers slow/fast convergence.
    - depth 6-10 tests shallow vs. moderately deep trees.
    - l2_leaf_reg provides regularisation against overfitting on minority class.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', CatBoostClassifier(silent=True, random_state=42)),
    ])

    param_grid = {
        'classifier__iterations': [200, 500, 800],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__depth': [6, 8, 10],
        'classifier__l2_leaf_reg': [1, 3, 5],
    }

    return pipeline, param_grid