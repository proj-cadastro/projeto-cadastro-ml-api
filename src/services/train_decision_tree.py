import os
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.utils.load_data import load_professores
from src.services.preprocessing import preprocess_for_decision_tree
from src.utils.logger import train_logger

MODELOS_DIR = "modelos_treinados"

def train_decision_tree_models():
    os.makedirs(MODELOS_DIR, exist_ok=True)
    df = load_professores()
    processed = preprocess_for_decision_tree(df, save_dir=MODELOS_DIR)

    for target, data in processed.items():
        X = data["X"]
        y = data["y"]

        model = DecisionTreeClassifier()
        model.fit(X, y)

        model_path = os.path.join(MODELOS_DIR, f"{target}_model.pkl")
        joblib.dump(model, model_path)

        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

        cm_dir = os.path.join("docs", "confusion_matrix", "decision_tree")
        os.makedirs(cm_dir, exist_ok=True)
        cm_fig = os.path.join(cm_dir, f"confusion_matrix_{target}_decision_tree.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax)
        plt.title(f"Matriz de confusão: {target} (Decision Tree)")
        plt.tight_layout()
        plt.savefig(cm_fig)
        plt.close(fig)

        train_logger.info(f"Modelo Decision Tree treinado: {target}")