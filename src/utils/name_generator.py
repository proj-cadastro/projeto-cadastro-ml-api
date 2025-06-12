import random
from src.utils.load_data import load_professores
import unicodedata

def gerar_nome_completo():
    with open("src/resources/nomes_masculinos.txt", encoding="utf-8") as f1, \
         open("src/resources/nomes_femininos.txt", encoding="utf-8") as f2, \
         open("src/resources/sobrenomes.txt", encoding="utf-8") as f3:
        nomes = f1.read().splitlines() + f2.read().splitlines()
        sobrenomes = f3.read().splitlines()
    return f"{random.choice(nomes)} {random.choice(sobrenomes)}"

def remover_acentos(txt):
    return ''.join(
        c for c in unicodedata.normalize('NFD', txt)
        if unicodedata.category(c) != 'Mn'
    )

def gerar_email_unico(nome, extensao):
    base = remover_acentos(nome).replace(" ", ".").lower()
    emails_existentes = load_professores()["email"].tolist()
    candidato = f"{base}@{extensao}"
    i = 1
    while candidato in emails_existentes:
        candidato = f"{base}{i}@{extensao}"
        i += 1
    return candidato

def gerar_lattes(nome):
    return f"https://lattes.com.br/{nome.replace(' ', '').lower()}"
