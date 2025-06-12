import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.utils.load_data import load_professores
from src.services.preprocessing import preprocess_for_neural_network
from src.utils.logger import train_logger

MODELOS_DIR = "modelos_treinados"
CAMPOS = ["titulacao", "email_ext", "referencia", "statusAtividade"]

def train_neural_network_models():
    os.makedirs(MODELOS_DIR, exist_ok=True)
    df = load_professores()

    for campo in CAMPOS:
        X, y_cat, y_encoded, encoder = preprocess_for_neural_network(df, campo)
        joblib.dump(encoder, os.path.join(MODELOS_DIR, f"{campo}_nn_encoder.pkl"))

        X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(Dense(64, activation="relu", input_shape=(X.shape[1],)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(y_cat.shape[1], activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val), verbose=0)
        loss, acc = model.evaluate(X_val, y_val, verbose=0)

        model.save(os.path.join(MODELOS_DIR, f"{campo}_nn.h5"))

        with open(os.path.join(MODELOS_DIR, f"{campo}_nn_info.json"), "w") as f:
            json.dump({"accuracy": acc}, f)

        y_val_pred = model.predict(X_val, verbose=0)
        y_val_pred_labels = np.argmax(y_val_pred, axis=1)
        y_val_true_labels = np.argmax(y_val, axis=1)

        cm = confusion_matrix(y_val_true_labels, y_val_pred_labels)

        cm_dir = os.path.join("docs", "confusion_matrix", "neural_network")
        os.makedirs(cm_dir, exist_ok=True)
        cm_fig = os.path.join(cm_dir, f"confusion_matrix_{campo}_nn.png")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
        plt.xlabel("Verdadeiro")
        plt.ylabel("Predito")
        plt.title(f"Matriz de confusão: {campo}")
        plt.tight_layout()
        plt.savefig(cm_fig)
        plt.close()

        train_logger.info(f"Modelo Rede Neural treinado: {campo}")
