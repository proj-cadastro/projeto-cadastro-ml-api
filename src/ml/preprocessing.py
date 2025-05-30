import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def extract_email_extension(email: str) -> str:
    return email.split("@")[-1]

def preprocess_professores(df: pd.DataFrame, save_dir: str) -> dict:
    df = df.copy()
    df["email_ext"] = df["email"].apply(extract_email_extension)
    df.drop(columns=["email"], inplace=True)

    targets = ["titulacao", "email_ext", "referencia", "statusAtividade"]
    X = df.drop(columns=targets)

    os.makedirs(save_dir, exist_ok=True)

    processed = {}
    for target in targets:
        y = df[target]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        joblib.dump(le, os.path.join(save_dir, f"{target}_encoder.pkl"))
        processed[target] = {"X": X.copy(), "y": y_encoded}

    return processed

def preprocess_cursos(df: pd.DataFrame, save_dir: str) -> dict:
    df = df.copy()
    target = "modelo"
    y = df[target]
    X = pd.DataFrame([{} for _ in range(len(df))])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(le, os.path.join(save_dir, f"{target}_encoder.pkl"))

    return {"X": X, "y": y_encoded}
