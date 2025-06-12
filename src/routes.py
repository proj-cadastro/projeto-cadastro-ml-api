from fastapi import APIRouter, Depends
from src.schemas.professor import ProfessorPartialInput, ProfessorFullOutput
from src.controllers import predict_professor_full_controller, predict_professor_partial_controller
from src.auth.api_key import validate_api_key
from src.utils.model_check import garantir_modelos_treinados
from fastapi import APIRouter, Depends, HTTPException
from src.services.train_decision_tree import train_decision_tree_models
from src.services.train_nn import train_neural_network_models
from src.services.generate_model_reports import generate_report

router = APIRouter(
    dependencies=[
        Depends(garantir_modelos_treinados),
        Depends(validate_api_key)
    ]  
)

@router.post("/predict/full", response_model=ProfessorFullOutput)
def predict_full():
    return predict_professor_full_controller()

@router.post("/predict/partial", response_model=dict)
def predict_partial(data: ProfessorPartialInput):
    return predict_professor_partial_controller(data)

@router.post("/train", tags=["Admin"])
def train_models():
    try:
        train_decision_tree_models()
        train_neural_network_models()
        generate_report()
        return {"detail": "Modelos treinados e relatórios gerados com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao treinar modelos: {str(e)}")