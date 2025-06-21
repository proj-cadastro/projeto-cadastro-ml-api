import os
from fastapi import HTTPException
from src.services.train_decision_tree import train_decision_tree_models
from src.services.train_nn import train_neural_network_models
from src.services.train_partial_models import train_partial_models
from src.services.generate_model_reports import generate_report

CAMPOS_COMPLETOS = ["titulacao", "email_ext", "referencia", "statusAtividade"]
CAMPOS_PARCIAIS = ["titulacao", "referencia", "statusAtividade"]

def modelos_treinados():
    # Modelos completos
    for campo in CAMPOS_COMPLETOS:
        dt_model = f"modelos_treinados/{campo}_model.pkl"
        nn_model = f"modelos_treinados/{campo}_nn.h5"
        if not (os.path.exists(dt_model) and os.path.exists(nn_model)):
            return False
    # Modelos parciais (6 modelos)
    partial_dir = "modelos_treinados/partial"
    nomes = [
        "parcial_titulacao.pkl",
        "parcial_referencia.pkl",
        "parcial_statusAtividade.pkl",
        "parcial_titulacao_referencia.pkl",
        "parcial_titulacao_statusAtividade.pkl",
        "parcial_referencia_statusAtividade.pkl"
    ]
    for nome in nomes:
        if not os.path.exists(os.path.join(partial_dir, nome)):
            return False
    return True

def garantir_modelos_treinados():
    if not modelos_treinados():
        try:
            train_decision_tree_models()
            train_neural_network_models()
            train_partial_models()
            generate_report()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao treinar modelos: {str(e)}")