import pandas as pd
from sklearn.model_selection import train_test_split

def load_ulb_data():
    # Carregar dados ULB 2013
    df = pd.read_csv('datasets/creditcard_2013.csv')
    
    # Verificar se há valores ausentes
    assert df.isnull().sum().sum() == 0, "Dataset contém valores ausentes."
    
    # Separar features e target
    X = df.drop(columns='Class')
    y = df['Class']
    
    # Divisão de dados (80% treino, 20% teste - estratificado)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    return X_train, X_test, y_train, y_test