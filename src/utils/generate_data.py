from src.utils.name_generator import gerar_nome_completo, gerar_email_unico, gerar_lattes

def gerar_nome_email_lattes(extensao_email):
    nome = gerar_nome_completo()
    email = gerar_email_unico(nome, extensao_email)
    lattes = gerar_lattes(nome)
    return nome, email, lattes