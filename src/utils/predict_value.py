import numpy as np
import pandas as pd
from src.utils.model_selector import get_best_model_type
from src.utils.model_loader import load_model_and_encoder
from src.utils.load_data import load_professores  # ADICIONE ESTA IMPORTAÇÃO

def predict_field_value(campo: str, X: pd.DataFrame) -> str:
    tipo = get_best_model_type(campo)
    model, encoder = load_model_and_encoder(campo, tipo)

    if tipo == "decision_tree":
        for col in model.feature_names_in_:
            if col not in X.columns:
                X[col] = 0
        X = X[model.feature_names_in_]
        pred = model.predict(X)
        return encoder.inverse_transform(pred)[0]

    elif tipo == "neural_network":
        
        df_treino = load_professores()
        df_treino = df_treino.copy()
        df_treino["email_ext"] = df_treino["email"].apply(lambda x: x.split("@")[-1] if "@" in x else "")
        df_treino.drop(columns=["email"], inplace=True)
        X_treino = pd.get_dummies(df_treino.drop(columns=["titulacao", "email_ext", "referencia", "statusAtividade"]))
        colunas_treino = sorted(X_treino.columns)
        X = X.reindex(columns=colunas_treino, fill_value=0)
        pred_probs = model.predict(X, verbose=0)
        pred_idx = np.argmax(pred_probs, axis=1)[0]
        return encoder.inverse_transform([pred_idx])[0]