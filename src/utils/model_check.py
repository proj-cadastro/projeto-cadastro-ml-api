import os
from fastapi import HTTPException
from src.services.train_decision_tree import train_decision_tree_models
from src.services.train_nn import train_neural_network_models
from src.services.generate_model_reports import generate_report

CAMPOS = ["titulacao", "email_ext", "referencia", "statusAtividade"]

def modelos_treinados():
    for campo in CAMPOS:
        dt_model = f"modelos_treinados/{campo}_model.pkl"
        nn_model = f"modelos_treinados/{campo}_nn.h5"
        if not (os.path.exists(dt_model) and os.path.exists(nn_model)):
            return False
    return True

def garantir_modelos_treinados():
    if not modelos_treinados():
        try:
            train_decision_tree_models()
            train_neural_network_models()
            generate_report()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao treinar modelos: {str(e)}")