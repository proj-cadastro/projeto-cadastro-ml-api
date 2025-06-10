from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from src.api.routes import router
from treinar import main as executar_treinamento

app = FastAPI()
app.include_router(router)

scheduler = BackgroundScheduler()
scheduler.add_job(executar_treinamento, "interval", hours=24)
scheduler.start()
