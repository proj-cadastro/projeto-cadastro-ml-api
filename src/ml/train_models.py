import os
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

def train_professor_models(processed: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    for target, data in processed.items():
        X = pd.get_dummies(data["X"])
        y = data["y"]

        model = DecisionTreeClassifier()
        model.fit(X, y)

        path = os.path.join(save_dir, f"{target}_model.pkl")
        joblib.dump(model, path)
