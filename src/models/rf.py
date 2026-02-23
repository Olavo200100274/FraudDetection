from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def get_pipeline_and_params(preprocessor):
    """Return (pipeline, param_grid) for Random Forest.

    Grid rationale (ULB 2013 / tabular fraud):
    - n_estimators ≥ 100 needed for stable ensembles on 228 K samples.
    - max_depth None allows full growth (common best for RF).
    - min_samples_leaf controls overfitting on the tiny fraud class.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42)),
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
    }

    return pipeline, param_grid