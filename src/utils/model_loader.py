import joblib
from tensorflow.keras.models import load_model

def load_model_and_encoder(campo: str, tipo: str):
    if tipo == "decision_tree":
        model = joblib.load(f"modelos_treinados/{campo}_model.pkl")
        encoder = joblib.load(f"modelos_treinados/{campo}_encoder.pkl")
    elif tipo == "neural_network":
        model = load_model(f"modelos_treinados/{campo}_nn.h5")
        encoder = joblib.load(f"modelos_treinados/{campo}_nn_encoder.pkl")
    else:
        raise ValueError(f"Tipo de modelo inválido: {tipo}")
    return model, encoder
