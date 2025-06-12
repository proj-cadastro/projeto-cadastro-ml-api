from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
import os

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="apikey")

def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Chave de API inválida")
    return api_key
