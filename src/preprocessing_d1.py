import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

def build_preprocessor_d1(log_amount: bool = True):
    # D1 columns: Time, V1..V28, Amount
    def add_log_amount(X):
        X = X.copy()
        if "Amount" in X.columns:
            X["Amount"] = np.log1p(X["Amount"].astype(float))
        return X

    steps = []
    if log_amount:
        steps.append(("log_amount", FunctionTransformer(add_log_amount, feature_names_out="one-to-one")))

    # scale all numeric columns (Time, V's, Amount)
    # scaling V's isn't strictly required but harmless and keeps consistency
    steps.append(("scaler", StandardScaler()))

    pre = Pipeline(steps=steps)
    return pre
