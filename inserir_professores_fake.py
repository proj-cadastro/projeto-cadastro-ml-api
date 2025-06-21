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

def ler_nomes_arquivo(caminho):
    with open(caminho, encoding="utf-8") as f:
        return [linha.strip() for linha in f if linha.strip()]

BASE_DIR = os.path.join(os.path.dirname(__file__), "src", "resources")
NOMES_FEMININOS = ler_nomes_arquivo(os.path.join(BASE_DIR, "nomes_femininos.txt"))
NOMES_MASCULINOS = ler_nomes_arquivo(os.path.join(BASE_DIR, "nomes_masculinos.txt"))
SOBRENOMES = ler_nomes_arquivo(os.path.join(BASE_DIR, "sobrenomes.txt"))

UNIDADES = ["003", "301", "132", "131", "174", "178", "112", "265", "299"]
REFERENCIAS = [
    "PES_I_A", "PES_II_A", "PES_III_A",
    "PES_I_B", "PES_II_B", "PES_III_B",
    "PES_I_C", "PES_II_C", "PES_III_C",
    "PES_I_D", "PES_II_D", "PES_III_D",
    "PES_I_E", "PES_II_E", "PES_III_E",
    "PES_I_F", "PES_II_F", "PES_III_F",
    "PES_I_G", "PES_II_G", "PES_III_G",
    "PES_I_H", "PES_II_H", "PES_III_H"
]
STATUS = ["ATIVO", "AFASTADO", "LICENCA", "NAO_ATIVO"]

TITULACAO_DISTRIB = {
    "DOUTOR": 0.15,
    "MESTRE": 0.60,
    "ESPECIALISTA": 0.25
}
EMAIL_DISTRIB = {
    "@fatec.sp.gov.br": 0.9,
    "@gmail.com": 0.1
}

# Distribuição de referencia por titulação (percentual cumulativo)
# Quanto maior o índice, mais difícil de aparecer
REFERENCIA_DISTRIB = {
    "DOUTOR": [
        (0.05, 26),  # 5% PES_III_H (mais alto)
        (0.15, 23),  # 10% PES_III_G
        (0.30, 20),  # 15% PES_III_F
        (0.50, 17),  # 20% PES_III_E
        (0.70, 14),  # 20% PES_III_D
        (0.85, 11),  # 15% PES_III_C
        (0.95, 8),   # 10% PES_III_B
        (1.00, 5)    # 5% PES_III_A
    ],
    "MESTRE": [
        (0.05, 26),  # 5% PES_III_H
        (0.15, 23),  # 10% PES_III_G
        (0.30, 20),  # 15% PES_III_F
        (0.50, 17),  # 20% PES_III_E
        (0.70, 14),  # 20% PES_III_D
        (0.85, 11),  # 15% PES_III_C
        (0.95, 8),   # 10% PES_III_B
        (1.00, 5)    # 5% PES_III_A
    ],
    "ESPECIALISTA": [
        (0.10, 8),   # 10% PES_III_B
        (0.30, 5),   # 20% PES_III_A
        (0.60, 2),   # 30% PES_II_C
        (0.90, 0),   # 30% PES_I_A
        (1.00, 1)    # 10% PES_II_A
    ]
}

def gerar_nome_completo():
    if random.random() < 0.5:
        nome = random.choice(NOMES_FEMININOS)
    else:
        nome = random.choice(NOMES_MASCULINOS)
    sobrenome = random.choice(SOBRENOMES)
    return f"{nome} {sobrenome}"

def gerar_email(nome, dominio):
    base = nome.lower().replace(" ", "")
    return f"{base}{random.randint(100,999)}{dominio}"

def gerar_lattes():
    return f"https://lattes.cnpq.br/{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=16))}"

def gerar_observacao():
    return random.choice(["", "Professor em avaliação", "Liberado para alocação", "Experiência internacional"])

def sortear_referencia(titulacao):
    dist = REFERENCIA_DISTRIB[titulacao]
    r = random.random()
    for p, idx in dist:
        if r <= p:
            if idx >= len(REFERENCIAS):
                idx = len(REFERENCIAS) - 1
            return REFERENCIAS[idx]
    return REFERENCIAS[0]

def inserir_professores(qtd=None):
    if qtd is None:
        qtd = int(os.getenv("QTD_PROFESSORES", 400))
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    qtd_doutor = int(qtd * TITULACAO_DISTRIB["DOUTOR"])
    qtd_mestre = int(qtd * TITULACAO_DISTRIB["MESTRE"])
    qtd_especialista = qtd - qtd_doutor - qtd_mestre

    titulacoes = (["DOUTOR"] * qtd_doutor +
                  ["MESTRE"] * qtd_mestre +
                  ["ESPECIALISTA"] * qtd_especialista)
    random.shuffle(titulacoes)

    qtd_fatec = int(qtd * EMAIL_DISTRIB["@fatec.sp.gov.br"])
    qtd_gmail = qtd - qtd_fatec

    dominios = (["@fatec.sp.gov.br"] * qtd_fatec +
                ["@gmail.com"] * qtd_gmail)
    random.shuffle(dominios)

    for i in range(qtd):
        nome = gerar_nome_completo()
        email = gerar_email(nome, dominios[i])
        titulacao = titulacoes[i]
        id_unidade = random.choice(UNIDADES)
        referencia = sortear_referencia(titulacao)
        lattes = gerar_lattes()
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
    inserir_professores()