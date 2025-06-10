import os
import json
from src.data.load_data import load_professores
from src.ml.preprocessing import preprocess_professores
from src.ml.train_models import train_professor_models
from src.ml.train_all_nn import main as train_all_nn
from src.ml.generate_model_reports import generate_report

CONTAGEM_PATH = "modelos_treinados/.count.json"
MODELOS_DIR = "modelos_treinados"

def get_registro_anterior():
    if not os.path.exists(CONTAGEM_PATH):
        return 0
    with open(CONTAGEM_PATH, "r") as f:
        data = json.load(f)
        return data.get("count", 0)

def salvar_nova_contagem(n):
    with open(CONTAGEM_PATH, "w") as f:
        json.dump({"count": n}, f)

def main():
    df = load_professores()
    total = len(df)
    anterior = get_registro_anterior()

    if total - anterior < 5:
        print(f"Apenas {total - anterior} novos registros. Nenhum modelo será re-treinado.")
        return

    print("Iniciando re-treinamento...")

    processed = preprocess_professores(df, save_dir=MODELOS_DIR)
    train_professor_models(processed, save_dir=MODELOS_DIR)
    train_all_nn()
    generate_report()
    salvar_nova_contagem(total)

    print("Modelos re-treinados com sucesso.")

if __name__ == "__main__":
    main()
