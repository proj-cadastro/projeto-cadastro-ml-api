from fastapi import HTTPException
from src.services.generate_model_reports import generate_report
from src.services.train_decision_tree import train_decision_tree_models
from src.services.train_nn import train_neural_network_models
from src.schemas.professor import ProfessorPartialInput, ProfessorFullOutput
from src.services.prediction import predict_professor_full, predict_professor_partial
from src.utils.logger import prediction_logger, error_logger

def predict_professor_full_controller():
    try:
        resultado = predict_professor_full()
        prediction_logger.info("Predição completa realizada com sucesso")
        return resultado
    except Exception as e:
        error_logger.error(f"Erro na predição completa: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno na predição completa.")

def predict_professor_partial_controller(data: ProfessorPartialInput):
    try:
        resultado = predict_professor_partial(data)
        prediction_logger.info("Predição parcial realizada com sucesso")
        return resultado
    except Exception as e:
        error_logger.error(f"Erro na predição parcial: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno na predição parcial.")

def train_models():
    try:
        train_decision_tree_models()
        train_neural_network_models()
        generate_report()
        return {"detail": "Modelos treinados e relatórios gerados com sucesso."}
    except Exception as e:
        error_logger.error(f"Erro ao treinar modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao treinar modelos: {str(e)}")