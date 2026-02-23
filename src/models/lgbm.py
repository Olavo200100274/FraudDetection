import lightgbm as lgb
from sklearn.pipeline import Pipeline


def get_pipeline_and_params(preprocessor):
    """Return (pipeline, param_grid) for LightGBM.

    Grid rationale (ULB 2013 / tabular fraud):
    - n_estimators 100-300 gives enough boosting rounds for convergence.
    - num_leaves controls tree complexity; 31 is default, 63 allows richer splits.
    - min_child_samples prevents overfitting on rare fraud class.
    - reg_alpha / reg_lambda add L1/L2 regularisation.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(random_state=42, verbosity=-1)),
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__num_leaves': [31, 63],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__min_child_samples': [20, 50],
        'classifier__reg_alpha': [0, 0.1],
        'classifier__reg_lambda': [0, 1],
    }

    return pipeline, param_grid