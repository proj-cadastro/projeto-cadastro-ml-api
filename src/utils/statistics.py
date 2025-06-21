import pandas as pd
import random
from src.utils.load_data import load_professores

REFERENCIA_ORDER = [
    "PES_I_A", "PES_II_A", "PES_III_A",
    "PES_I_B", "PES_II_B", "PES_III_B",
    "PES_I_C", "PES_II_C", "PES_III_C",
    "PES_I_D", "PES_II_D", "PES_III_D",
    "PES_I_E", "PES_II_E", "PES_III_E",
    "PES_I_F", "PES_II_F", "PES_III_F",
    "PES_I_G", "PES_II_G", "PES_III_G",
    "PES_I_H", "PES_II_H", "PES_III_H"
]

def get_top3_random(field):
    df = load_professores()
    top3 = df[field].value_counts().nlargest(3).index.tolist()
    return random.choice(top3) if top3 else None

def get_referencia_top5_random():
    df = load_professores()
    top5 = df["referencia"].value_counts().nlargest(3).index.tolist()
    return random.choice(top5) if top5 else None