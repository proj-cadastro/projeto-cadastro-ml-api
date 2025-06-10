import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from src.data.load_data import load_professores
from src.ml.preprocessing import extract_email_extension
from tensorflow.keras.models import load_model

def preprocess_for_classical(df, target):
    df = df.copy()
    df["email_ext"] = df["email"].apply(extract_email_extension)
    df.drop(columns=["email"], inplace=True)
    X = pd.get_dummies(df.drop(columns=["titulacao", "email_ext", "referencia", "statusAtividade"]))
    y = df[target]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded

def evaluate_decision_tree(X, y, model_path):
    model = joblib.load(model_path)
    X_filled = X.copy()
    for col in model.feature_names_in_:
        if col not in X_filled.columns:
            X_filled[col] = 0
    X_filled = X_filled[model.feature_names_in_]
    y_pred = model.predict(X_filled)
    return accuracy_score(y, y_pred)

def generate_report():
    campos = ["titulacao", "email_ext", "referencia", "statusAtividade"]
    df = load_professores()
    os.makedirs("docs", exist_ok=True)

    for campo in campos:
        X, y = preprocess_for_classical(df, campo)
        arv_acc = evaluate_decision_tree(X, y, f"modelos_treinados/{campo}_model.pkl")

        with open(f"modelos_treinados/{campo}_nn_info.json", "r") as f:
            nn_info = json.load(f)
        nn_acc = nn_info["accuracy"]

        melhor_modelo = "neural_network" if nn_acc >= arv_acc else "decision_tree"
        report_lines = [
            f"Campo predito: {campo}",
            f"Melhor modelo: {melhor_modelo}",
            f"Acurácia rede neural: {nn_acc:.4f}",
            f"Acurácia árvore de decisão: {arv_acc:.4f}",
            f"Modelo árvore: modelos_treinados/{campo}_model.pkl",
            f"Encoder árvore: modelos_treinados/{campo}_encoder.pkl",
            f"Modelo NN: modelos_treinados/{campo}_nn.h5",
            f"Encoder NN: modelos_treinados/{campo}_nn_encoder.pkl",
            f"Matriz de confusão NN: modelos_treinados/confusion_matrix_{campo}_nn.png"
        ]

        if melhor_modelo == "neural_network":
            model = load_model(f"modelos_treinados/{campo}_nn.h5")
            architecture = [layer.output_shape[1] for layer in model.layers if hasattr(layer, "output_shape")]
            report_lines.append(f"Arquitetura da rede neural (número de neurônios por camada): {architecture}")

        with open(f"docs/{campo}_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

if __name__ == "__main__":
    generate_report()
