from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import os
from dotenv import load_dotenv
from src.config.database import get_engine
from pathlib import Path

load_dotenv()

def load_professores():
    try:
        engine = get_engine()
        query = "SELECT titulacao, email, idUnidade, referencia, statusAtividade FROM professor"
        df = pd.read_sql(query, engine)
        return df
    except SQLAlchemyError:
        BASE_DIR = Path(__file__).resolve().parent.parent
        CSV_PATH = BASE_DIR / "resources" / "dataset" / "professores.csv"
        return pd.read_csv(CSV_PATH)