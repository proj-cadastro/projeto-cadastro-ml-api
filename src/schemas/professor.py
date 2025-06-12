from pydantic import BaseModel
from typing import Optional

class ProfessorPartialInput(BaseModel):
    titulacao: Optional[str] = None
    idUnidade: Optional[str] = None
    referencia: Optional[str] = None
    statusAtividade: Optional[str] = None
    nome: Optional[str] = None

class ProfessorFullOutput(BaseModel):
    nome: str
    email: str
    titulacao: str
    idUnidade: str
    referencia: str
    statusAtividade: str
    lattes: str
