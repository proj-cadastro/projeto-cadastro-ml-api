import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from src.utils.email import extract_email_extension

def preprocess_for_decision_tree(df: pd.DataFrame, save_dir: str) -> dict:
    df = df.copy()
    df["email_ext"] = df["email"].apply(extract_email_extension)
    df.drop(columns=["email"], inplace=True)
    os.makedirs(save_dir, exist_ok=True)

    targets = ["titulacao", "email_ext", "referencia", "statusAtividade"]
    X = pd.get_dummies(df.drop(columns=targets))

    result = {}
    for target in targets:
        y = df[target]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        joblib.dump(le, os.path.join(save_dir, f"{target}_encoder.pkl"))
        result[target] = {"X": X.copy(), "y": y_encoded}

    return result

def preprocess_for_neural_network(df: pd.DataFrame, target: str):
    df = df.copy()
    df["email_ext"] = df["email"].apply(extract_email_extension)
    df.drop(columns=["email"], inplace=True)

    X = pd.get_dummies(df.drop(columns=["titulacao", "email_ext", "referencia", "statusAtividade"]))
    y = df[target]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    return X, y_cat, y_encoded, le
