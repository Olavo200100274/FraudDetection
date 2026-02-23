from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline


def get_pipeline_and_params(preprocessor):
    """Return (pipeline, param_grid) for One-Class SVM.

    OCSVM is an anomaly-detection baseline:
    - Trained only on class-0 (legitimate) samples.
    - Uses negated decision_function for scoring (higher = more anomalous).
    - nu upper-bounds the fraction of training errors / support vectors;
      values close to the true fraud rate (≈0.17 %) are tested.
    - gamma controls the RBF kernel width.
    - No grid search (fixed config) because OCSVM does not support CV
      scoring in the standard supervised sense.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')),
    ])

    param_grid = None  # no grid search for OCSVM

    return pipeline, param_grid