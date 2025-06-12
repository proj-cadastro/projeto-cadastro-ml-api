import os
import json
from src.utils.load_data import load_professores

CONTAGEM_PATH = "modelos_treinados/.count.json"

def get_registro_anterior():
    if not os.path.exists(CONTAGEM_PATH):
        return 0
    with open(CONTAGEM_PATH, "r") as f:
        data = json.load(f)
        return data.get("count", 0)

def salvar_nova_contagem(n):
    with open(CONTAGEM_PATH, "w") as f:
        json.dump({"count": n}, f)

def precisa_retreinar():
    df = load_professores()
    total = len(df)
    anterior = get_registro_anterior()
    return (total - anterior) >= 5, total
