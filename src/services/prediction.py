import os
import random
import joblib
import pandas as pd
from src.api.schemas import ProfessorPartialInput, ProfessorFullOutput
from fastapi import HTTPException
from src.data.load_data import load_professores
from src.ml.preprocessing import extract_email_extension

MODELS_DIR = "modelos_treinados"

def load_model_and_encoder(name):
    model = joblib.load(os.path.join(MODELS_DIR, f"{name}_model.pkl"))
    encoder = joblib.load(os.path.join(MODELS_DIR, f"{name}_encoder.pkl"))
    return model, encoder

def gerar_nome_completo():
    with open("src/resources/nomes_masculinos.txt", encoding="utf-8") as f1, \
         open("src/resources/nomes_femininos.txt", encoding="utf-8") as f2, \
         open("src/resources/sobrenomes.txt", encoding="utf-8") as f3:
        nomes = f1.read().splitlines() + f2.read().splitlines()
        sobrenomes = f3.read().splitlines()
    return f"{random.choice(nomes)} {random.choice(sobrenomes)}"

def gerar_email_unico(nome, extensao, emails_existentes):
    base = nome.replace(" ", "").lower()
    candidato = f"{base}@{extensao}"
    i = 1
    while candidato in emails_existentes:
        candidato = f"{base}{i}@{extensao}"
        i += 1
    return candidato

def gerar_lattes(nome):
    return f"https://lattes.com.br/{nome.replace(' ', '').lower()}"

def predict_professor_full():
    try:
        dados = load_professores()

        X = pd.get_dummies(dados.drop(columns=["titulacao", "email", "referencia", "statusAtividade"]))
        emails_existentes = dados["email"].tolist()

        predicoes = {}
        for campo in ["titulacao", "email_ext", "referencia", "statusAtividade"]:
            try:
                model, encoder = load_model_and_encoder(campo)
                pred = model.predict(X.sample(1))
                valor = encoder.inverse_transform(pred)[0]
                predicoes[campo] = valor
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Erro ao prever '{campo}': {str(e)}")

        try:
            nome = gerar_nome_completo()
            email = gerar_email_unico(nome, predicoes["email_ext"], emails_existentes)
            lattes = gerar_lattes(nome)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao gerar nome, email ou lattes: {str(e)}")

        return ProfessorFullOutput(
            nome=nome,
            email=email,
            titulacao=predicoes["titulacao"],
            idUnidade=str(dados.iloc[0]["idUnidade"]),
            referencia=predicoes["referencia"],
            statusAtividade=predicoes["statusAtividade"],
            lattes=lattes
        )

    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno inesperado: {str(e)}")

def predict_professor_partial(data: ProfessorPartialInput):
    try:
        entrada = data.dict(exclude_none=True)
        X = pd.DataFrame([entrada])
        X = pd.get_dummies(X)

        previsoes = {}
        for campo in ["titulacao", "email_ext", "referencia", "statusAtividade"]:
            if campo not in entrada:
                try:
                    model, encoder = load_model_and_encoder(campo)

                    colunas_esperadas = model.feature_names_in_
                    for col in colunas_esperadas:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[colunas_esperadas]

                    pred = model.predict(X)
                    previsoes[campo] = encoder.inverse_transform(pred)[0]

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Erro ao prever '{campo}': {str(e)}")

        return previsoes

    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno inesperado: {str(e)}")
