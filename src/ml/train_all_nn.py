import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from src.data.load_data import load_professores
from src.ml.preprocessing import extract_email_extension

def preprocess(df, target):
    df = df.copy()
    df["email_ext"] = df["email"].apply(extract_email_extension)
    df.drop(columns=["email"], inplace=True)
    X = pd.get_dummies(df.drop(columns=["titulacao", "email_ext", "referencia", "statusAtividade"]))
    y = df[target]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded)
    return X, y_cat, y_encoded, le

def train_nn(X, y_cat, input_dim, output_dim):
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(input_dim,)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(output_dim, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def save_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 7))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    df = load_professores()
    save_dir = "modelos_treinados"
    os.makedirs(save_dir, exist_ok=True)
    campos = ["titulacao", "email_ext", "referencia", "statusAtividade"]

    for campo in campos:
        X, y_cat, y_encoded, encoder = preprocess(df, campo)
        joblib.dump(encoder, os.path.join(save_dir, f"{campo}_nn_encoder.pkl"))
        X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)
        model = train_nn(X_train, y_train, X.shape[1], y_cat.shape[1])
        history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val), verbose=0)
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        model.save(os.path.join(save_dir, f"{campo}_nn.h5"))
        with open(os.path.join(save_dir, f"{campo}_nn_info.json"), "w") as f:
            json.dump({"accuracy": acc}, f)
            
        y_val_pred = model.predict(X_val, verbose=0)
        y_val_pred_labels = np.argmax(y_val_pred, axis=1)
        y_val_true_labels = np.argmax(y_val, axis=1)

        save_confusion_matrix(
            y_val_true_labels,
            y_val_pred_labels,
            labels=encoder.classes_,
            save_path=os.path.join(save_dir, f"confusion_matrix_{campo}_nn.png")
        )

if __name__ == "__main__":
    main()
