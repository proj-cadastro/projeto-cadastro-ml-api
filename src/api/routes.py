from fastapi import APIRouter
from src.api.schemas import ProfessorPartialInput, ProfessorFullOutput
from src.services.prediction import predict_professor_full, predict_professor_partial

router = APIRouter()

@router.post("/predict/professor/full", response_model=ProfessorFullOutput)
def predict_full():
    return predict_professor_full()

@router.post("/predict/professor/partial", response_model=dict)
def predict_partial(data: ProfessorPartialInput):
    return predict_professor_partial(data)
