from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sklearn
sklearn.set_config(transform_output="pandas")

def get_preprocessor(X_train):
    # Identificar features numéricas
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Criar um pré-processador para as features numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Aplicar o preprocessor (verbose_feature_names_out=False keeps original names)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor