from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import os
from dotenv import load_dotenv
from src.config.database import get_engine
from pathlib import Path
from src.utils.logger import database_logger

load_dotenv()

def load_professores():
    try:
        if os.getenv("USE_CSV", "false").lower() == "true":
            raise SQLAlchemyError
        
        engine = get_engine()
        query = "SELECT titulacao, email, idUnidade, referencia, statusAtividade FROM professor"
        df = pd.read_sql(query, engine)
        database_logger.info("Banco de dados utilizado para carregar professores")
        return df
    except SQLAlchemyError:
        BASE_DIR = Path(__file__).resolve().parent.parent
        CSV_PATH = BASE_DIR / "resources" / "dataset" / "professores.csv"
        database_logger.info("CSV utilizado para carregar professores")
        return pd.read_csv(CSV_PATH)