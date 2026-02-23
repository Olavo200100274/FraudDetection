from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def get_pipeline_and_params(preprocessor):
    """Return (pipeline, param_grid) for Logistic Regression.

    Grid rationale (ULB 2013 / tabular fraud):
    - C spans 4 orders of magnitude to find the right regularisation strength.
    - Both L1 and L2 penalties are tested (L1 can zero-out irrelevant PCA dims).
    - 'saga' supports both penalties and scales well; 'liblinear' is fast for L1.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=5000, solver='saga', random_state=42)),
    ])

    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__l1_ratio': [0, 0.5, 1],   # 0=L2, 1=L1, 0.5=ElasticNet
    }

    return pipeline, param_grid