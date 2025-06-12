import numpy as np
import pandas as pd
from src.utils.model_selector import get_best_model_type
from src.utils.model_loader import load_model_and_encoder

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
        X = X.reindex(columns=sorted(X.columns), fill_value=0)
        pred_probs = model.predict(X, verbose=0)
        pred_idx = np.argmax(pred_probs, axis=1)[0]
        return encoder.inverse_transform([pred_idx])[0]
