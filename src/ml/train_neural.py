import os
import json
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def train_referencia_nn(df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    df = df.copy()
    df["email_ext"] = df["email"].apply(lambda e: e.split("@")[-1])
    df.drop(columns=["email"], inplace=True)

    target = "referencia"
    X = pd.get_dummies(df.drop(columns=["titulacao", "referencia", "statusAtividade"]))
    y = df[target]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    joblib.dump(le, os.path.join(save_dir, "referencia_nn_encoder.pkl"))

    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(X.shape[1],)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(y_cat.shape[1], activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val), verbose=0)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    model.save(os.path.join(save_dir, "referencia_nn.h5"))
    with open(os.path.join(save_dir, "referencia_nn_info.json"), "w") as f:
        json.dump({"accuracy": val_acc}, f)

    print(f"Rede neural treinada com acurácia: {val_acc:.4f}")

if __name__ == "__main__":
    df = pd.read_csv("src/data/professores.csv")
    train_referencia_nn(df, "modelos_treinados")
