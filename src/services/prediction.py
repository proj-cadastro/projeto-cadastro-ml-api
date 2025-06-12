import pandas as pd
from src.schemas.professor import ProfessorPartialInput, ProfessorFullOutput
from src.utils.load_data import load_professores
from src.utils.email import extract_email_extension
from src.utils.predict_value import predict_field_value
from src.utils.generate_data import gerar_nome_email_lattes
from src.utils.name_generator import gerar_email_unico, gerar_lattes

def predict_professor_full():
    df = load_professores()
    X = pd.get_dummies(df.drop(columns=["titulacao", "email", "referencia", "statusAtividade"]))
    X = X.reindex(columns=sorted(X.columns), fill_value=0)
    X_sample = X.sample(1)

    email_ext = predict_field_value("email_ext", X_sample)
    nome, email, lattes = gerar_nome_email_lattes(email_ext)

    return ProfessorFullOutput(
        nome=nome,
        email=email,
        titulacao=predict_field_value("titulacao", X_sample),
        idUnidade=str(df.iloc[0]["idUnidade"]),
        referencia=predict_field_value("referencia", X_sample),
        statusAtividade=predict_field_value("statusAtividade", X_sample),
        lattes=lattes
    )

def predict_professor_partial(data: ProfessorPartialInput):
    entrada = data.model_dump(exclude_none=True)
    df = pd.DataFrame([entrada])

    if "email" in df.columns:
        df["email_ext"] = df["email"].apply(extract_email_extension)
        df.drop(columns=["email"], inplace=True)

    X = pd.get_dummies(df)
    X = X.reindex(columns=sorted(X.columns), fill_value=0)

    campos_alvo = ["titulacao", "email_ext", "referencia", "statusAtividade"]
    previsoes = {}

    for campo in campos_alvo:
        if campo not in entrada:
            previsoes[campo] = predict_field_value(campo, X)

    # Geração de nome, email e lattes
    if "nome" not in entrada:
        email_ext = previsoes.get("email_ext", entrada.get("email_ext"))
        nome, _, _ = gerar_nome_email_lattes(email_ext)
        previsoes["nome"] = nome
        previsoes["email"] = gerar_email_unico(nome, email_ext)
        previsoes["lattes"] = gerar_lattes(nome)
    else:
        email_ext = previsoes.get("email_ext", entrada.get("email_ext"))
        nome = entrada["nome"]
        previsoes["email"] = gerar_email_unico(nome, email_ext)
        previsoes["lattes"] = gerar_lattes(nome)

    ordem = [
        "nome",
        "email",
        "titulacao",
        "idUnidade",
        "email_ext",
        "referencia",
        "statusAtividade",
        "lattes"
    ]
    resposta = {}
    for campo in ordem:
        if campo not in entrada and campo in previsoes:
            resposta[campo] = previsoes[campo]
    return resposta