from pydantic import BaseModel
from typing import Optional

class ProfessorPartialInput(BaseModel):
    titulacao: Optional[str]
    idUnidade: Optional[str]
    referencia: Optional[str]
    statusAtividade: Optional[str]

class ProfessorFullOutput(BaseModel):
    nome: str
    email: str
    titulacao: str
    idUnidade: str
    referencia: str
    statusAtividade: str
    lattes: str
