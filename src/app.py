from fastapi import FastAPI
from src.routes import router
from src.services.train_decision_tree import train_decision_tree_models
from src.services.train_nn import train_neural_network_models
from src.services.generate_model_reports import generate_report
from src.utils.retrain_condition import precisa_retreinar, salvar_nova_contagem
from src.utils.logger import train_logger

app = FastAPI()
app.include_router(router)

def agendamento():
    precisa, total = precisa_retreinar()
    if precisa:
        try:
            train_logger.info("Iniciando re-treinamento agendado")
            train_decision_tree_models()
            train_neural_network_models()
            generate_report()
            salvar_nova_contagem(total)
            train_logger.info("Re-treinamento finalizado com sucesso")
        except Exception as e:
            train_logger.error(f"Erro durante re-treinamento agendado: {str(e)}")