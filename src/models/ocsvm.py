from sklearn.svm import OneClassSVM

def build_model():
    return OneClassSVM(
        kernel="rbf",
        nu=0.01,
        gamma="scale",
    )
