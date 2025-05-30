import os
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

def train_professor_models(processed: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    for target, data in processed.items():
        if target == "curso_modelo":
            continue  # pula o curso

        X = pd.get_dummies(data["X"])
        y = data["y"]

        model = DecisionTreeClassifier()
        model.fit(X, y)

        path = os.path.join(save_dir, f"{target}_model.pkl")
        joblib.dump(model, path)

def train_curso_model(processed: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    data = processed["curso_modelo"]
    y = data["y"]

    # Como não temos features ainda, usamos DummyClassifier
    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0]] * len(y), y)  # lista de "falsas features"

    path = os.path.join(save_dir, "curso_modelo_model.pkl")
    joblib.dump(model, path)
