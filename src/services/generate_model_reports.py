import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from tensorflow.keras.models import load_model
from src.utils.load_data import load_professores
from src.services.preprocessing import preprocess_for_decision_tree, preprocess_for_neural_network
from src.utils.email import extract_email_extension

def evaluate_tree_model(X, y, campo):
    model_path = f"modelos_treinados/{campo}_model.pkl"
    model = joblib.load(model_path)
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0
    X = X[model.feature_names_in_]
    y_pred = model.predict(X)
    return {
        "type": "decision_tree",
        "model": model_path,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "confusion": confusion_matrix(y, y_pred).tolist()
    }

def evaluate_nn_model(X, y_cat, y_true, campo):
    model_path = f"modelos_treinados/{campo}_nn.h5"
    model = load_model(model_path)
    y_pred_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    architecture = [layer.output_shape[1] for layer in model.layers if hasattr(layer, "output_shape")]
    return {
        "type": "neural_network",
        "model": model_path,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
        "architecture": architecture
    }

def write_model_report(campo, tipo, dados):
    path_dir = f"docs/{'neural_network' if tipo == 'neural_network' else 'decision_tree'}"
    os.makedirs(path_dir, exist_ok=True)
    path = os.path.join(path_dir, f"{campo}.txt")

    lines = []
    if tipo == "neural_network":
        lines = [
            f"Rede Neural:",
            f"Arquitetura do Modelo: {dados['architecture']}",
            f"Hiperparâmetros: Otimizador: Adam, Função de Ativação: ReLU, Épocas: 50, Batch: 8",
            f"Desempenho: Acurácia: {dados['accuracy']:.4f}, Precisão: {dados['precision']:.4f}, Recall: {dados['recall']:.4f}, F1: {dados['f1']:.4f}",
            f"Análise de Overfitting/Underfitting: Verificar matriz de confusão",
            f"Tempo de Treinamento: (Não registrado)",
            f"Matriz de Confusão: {dados['confusion']}"
        ]
    else:
        lines = [
            f"Árvore de Decisão:",
            f"Características do Modelo: Tipo: DecisionTreeClassifier",
            f"Desempenho: Acurácia: {dados['accuracy']:.4f}, Precisão: {dados['precision']:.4f}, Recall: {dados['recall']:.4f}, F1: {dados['f1']:.4f}",
            f"Importância das Variáveis: (não extraído)",
            f"Análise de Overfitting/Underfitting: Comparar erro de treino/teste",
            f"Tempo de Treinamento: (Não registrado)",
            f"Matriz de Confusão: {dados['confusion']}"
        ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def generate_report():
    campos = ["titulacao", "email_ext", "referencia", "statusAtividade"]
    df = load_professores()
    comparativo = []

    for campo in campos:
        dados_tree = preprocess_for_decision_tree(df, save_dir="modelos_treinados")[campo]
        X_tree, y_tree = dados_tree["X"], dados_tree["y"]
        arv = evaluate_tree_model(X_tree, y_tree, campo)
        write_model_report(campo, "decision_tree", arv)

        X_nn, y_cat, y_encoded, _ = preprocess_for_neural_network(df, campo)
        nn = evaluate_nn_model(X_nn, y_cat, y_encoded, campo)
        write_model_report(campo, "neural_network", nn)

        melhor = "neural_network" if nn["accuracy"] >= arv["accuracy"] else "decision_tree"
        comparativo.append(
            f"{campo}:\n  Melhor modelo: {melhor}\n  Acurácia NN: {nn['accuracy']:.4f}\n  Acurácia DT: {arv['accuracy']:.4f}"
        )

    with open("docs/comparativo.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(comparativo))

if __name__ == "__main__":
    generate_report()
