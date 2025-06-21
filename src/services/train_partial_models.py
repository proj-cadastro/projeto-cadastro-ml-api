import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.utils.load_data import load_professores

MODELOS_DIR = "modelos_treinados/partial"
os.makedirs(MODELOS_DIR, exist_ok=True)

def train_partial_models():
    df = load_professores()
    campos = ["titulacao", "referencia", "statusAtividade"]

    # Modelos para 1 campo faltante
    for target in campos:
        features = [c for c in campos if c != target]
        X = pd.get_dummies(df[features])
        y = df[target]
        model = DecisionTreeClassifier()
        model.fit(X, y)
        joblib.dump(model, os.path.join(MODELOS_DIR, f"parcial_{target}.pkl"))

    # Modelos para 2 campos faltantes
    for i, target1 in enumerate(campos):
        for target2 in campos[i+1:]:
            feature = [c for c in campos if c not in [target1, target2]][0]
            X = pd.get_dummies(df[[feature]])
            y = df[[target1, target2]]
            model = DecisionTreeClassifier()
            model.fit(X, y)
            joblib.dump(model, os.path.join(MODELOS_DIR, f"parcial_{target1}_{target2}.pkl"))

if __name__ == "__main__":
    train_partial_models()