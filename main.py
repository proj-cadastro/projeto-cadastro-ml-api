from src.app import agendamento, app
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(agendamento, "interval", hours=2)
scheduler.start()

app = app