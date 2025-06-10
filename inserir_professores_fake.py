import random
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS") or "",
    "database": os.getenv("DB_NAME")
}

NOMES = [
    "Ana", "João", "Carlos", "Fernanda", "Beatriz", "Lucas", "Juliana",
    "Mateus", "Camila", "Eduardo", "Marina", "Vinicius", "Larissa", "Pedro"
]
SOBRENOMES = [
    "Silva", "Souza", "Oliveira", "Costa", "Pereira", "Rodrigues", "Almeida", "Barbosa"
]

TITULACOES = ["DOUTOR", "MESTRE", "ESPECIALISTA"]
REFERENCIAS = [f"PES_{nivel}_{letra}" for nivel in ["I", "II", "III"] for letra in "ABCDEFGH"]
STATUS = ["ATIVO", "AFASTADO", "LICENCA", "NAO_ATIVO"]

def gerar_nome_completo():
    return f"{random.choice(NOMES)} {random.choice(SOBRENOMES)}"

def gerar_email(nome):
    base = nome.lower().replace(" ", "")
    dominio = random.choice(["gmail.com", "hotmail.com", "yahoo.com.br", "fatec.sp.gov.br"])
    return f"{base}@{dominio}"

def gerar_id_unidade():
    return str(random.randint(1000, 1099))

def gerar_lattes(nome):
    return f"https://lattes.com.br/{nome.replace(' ', '').lower()}"

def gerar_observacao():
    return random.choice(["", "Professor em avaliação", "Liberado para alocação", "Experiência internacional"])

def inserir_professores(qtd=30):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    for _ in range(qtd):
        nome = gerar_nome_completo()
        email = gerar_email(nome)
        titulacao = random.choice(TITULACOES)
        id_unidade = gerar_id_unidade()
        referencia = random.choice(REFERENCIAS)
        lattes = gerar_lattes(nome)
        status = random.choice(STATUS)
        observacoes = gerar_observacao()

        cursor.execute("""
            INSERT INTO professor
                (nome, email, titulacao, idUnidade, referencia, lattes, statusAtividade, observacoes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (nome, email, titulacao, id_unidade, referencia, lattes, status, observacoes))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"{qtd} professores inseridos com sucesso.")

if __name__ == "__main__":
    inserir_professores(30)
