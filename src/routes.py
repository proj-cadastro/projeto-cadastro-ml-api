from fastapi import APIRouter, Depends
from src.schemas.professor import ProfessorPartialInput, ProfessorFullOutput
from src.controllers import predict_professor_full_controller, predict_professor_partial_controller
from src.auth.api_key import validate_api_key
from src.utils.model_check import garantir_modelos_treinados
from fastapi import HTTPException
from src.controllers import train_models

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
def train_models_route():
    return train_models()