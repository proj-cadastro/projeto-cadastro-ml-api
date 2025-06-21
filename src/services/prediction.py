import pandas as pd
import joblib
from src.schemas.professor import ProfessorPartialInput, ProfessorFullOutput
from src.utils.load_data import load_professores
from src.utils.email import extract_email_extension
from src.utils.predict_value import predict_field_value
from src.utils.generate_data import gerar_nome_email_lattes
from src.utils.name_generator import gerar_email_unico, gerar_lattes
from src.utils.statistics import get_top3_random, get_referencia_top5_random

PARTIAL_MODELS_DIR = "modelos_treinados/partial"

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
    campos = ["titulacao", "referencia", "statusAtividade"]
    presentes = [c for c in campos if c in entrada]
    faltantes = [c for c in campos if c not in entrada]

    if "nome" not in entrada:
        nome, _, _ = gerar_nome_email_lattes("fatec.sp.gov.br")
        entrada["nome"] = nome
    if "email" not in entrada:
        entrada["email"] = gerar_email_unico(entrada["nome"], "fatec.sp.gov.br")
    if "lattes" not in entrada:
        entrada["lattes"] = gerar_lattes(entrada["nome"])

    previsoes = {}

    if 1 <= len(faltantes) <= 2 and len(presentes) > 0:
        model_name = "parcial_" + "_".join(sorted(faltantes)) + ".pkl"
        model_path = f"{PARTIAL_MODELS_DIR}/{model_name}"
        model = joblib.load(model_path)
        X = pd.get_dummies(pd.DataFrame([{c: entrada[c] for c in presentes}]))
        for col in model.feature_names_in_:
            if col not in X.columns:
                X[col] = 0
        X = X[model.feature_names_in_]
        pred = model.predict(X)
        if len(faltantes) == 1:
            previsoes[faltantes[0]] = pred[0]
        else:
            previsoes[faltantes[0]] = pred[0][0]
            previsoes[faltantes[1]] = pred[0][1]
    else:
        if "titulacao" not in entrada:
            previsoes["titulacao"] = get_top3_random("titulacao")
        if "referencia" not in entrada:
            previsoes["referencia"] = get_referencia_top5_random()
        if "statusAtividade" not in entrada:
            previsoes["statusAtividade"] = get_top3_random("statusAtividade")

    ordem = [
        "nome",
        "email",
        "titulacao",
        "idUnidade",
        "referencia",
        "statusAtividade",
        "lattes"
    ]
    resposta = {}
    for campo in ordem:
        if campo in entrada:
            resposta[campo] = entrada[campo]
        elif campo in previsoes:
            resposta[campo] = previsoes[campo]
    return resposta