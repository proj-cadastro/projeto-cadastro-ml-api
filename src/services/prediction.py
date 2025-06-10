import os
import random
import joblib
import json
import numpy as np
import pandas as pd
from fastapi import HTTPException
from tensorflow.keras.models import load_model
from src.api.schemas import ProfessorPartialInput, ProfessorFullOutput
from src.data.load_data import load_professores
from src.ml.preprocessing import extract_email_extension

MODELS_DIR = "modelos_treinados"
DOCS_DIR = "docs"

def get_best_model_type(campo):
    path = os.path.join(DOCS_DIR, f"{campo}_report.txt")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("Melhor modelo:"):
                return line.strip().split(":")[1].strip()
    raise HTTPException(status_code=500, detail=f"Não foi possível determinar o modelo para o campo '{campo}'.")

def load_model_and_encoder(campo):
    tipo = get_best_model_type(campo)
    if tipo == "decision_tree":
        modelo = joblib.load(os.path.join(MODELS_DIR, f"{campo}_model.pkl"))
        encoder = joblib.load(os.path.join(MODELS_DIR, f"{campo}_encoder.pkl"))
        return ("decision_tree", modelo, encoder)
    elif tipo == "neural_network":
        modelo = load_model(os.path.join(MODELS_DIR, f"{campo}_nn.h5"))
        encoder = joblib.load(os.path.join(MODELS_DIR, f"{campo}_nn_encoder.pkl"))
        return ("neural_network", modelo, encoder)
    else:
        raise HTTPException(status_code=500, detail=f"Tipo de modelo inválido para '{campo}': {tipo}")

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

def predict_value(campo, X_input):
    tipo, modelo, encoder = load_model_and_encoder(campo)
    if tipo == "decision_tree":
        for col in modelo.feature_names_in_:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[modelo.feature_names_in_]
        pred = modelo.predict(X_input)
        return encoder.inverse_transform(pred)[0]
    elif tipo == "neural_network":
        modelo_input = X_input.copy()
        modelo_input = modelo_input.reindex(columns=sorted(modelo_input.columns), fill_value=0)
        pred_probs = modelo.predict(modelo_input, verbose=0)
        pred_idx = np.argmax(pred_probs, axis=1)[0]
        return encoder.inverse_transform([pred_idx])[0]

def predict_professor_full():
    try:
        dados = load_professores()
        X = pd.get_dummies(dados.drop(columns=["titulacao", "email", "referencia", "statusAtividade"]))
        X = X.reindex(columns=sorted(X.columns), fill_value=0)
        X_sample = X.sample(1)
        emails_existentes = dados["email"].tolist()

        predicoes = {}
        for campo in ["titulacao", "email_ext", "referencia", "statusAtividade"]:
            predicoes[campo] = predict_value(campo, X_sample)

        nome = gerar_nome_completo()
        email = gerar_email_unico(nome, predicoes["email_ext"], emails_existentes)
        lattes = gerar_lattes(nome)

        return ProfessorFullOutput(
            nome=nome,
            email=email,
            titulacao=predicoes["titulacao"],
            idUnidade=str(dados.iloc[0]["idUnidade"]),
            referencia=predicoes["referencia"],
            statusAtividade=predicoes["statusAtividade"],
            lattes=lattes
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição full: {str(e)}")

def predict_professor_partial(data: ProfessorPartialInput):
    try:
        entrada = data.model_dump(exclude_none=True)
        X = pd.DataFrame([entrada])
        X["email_ext"] = X["email"].apply(extract_email_extension) if "email" in X else None
        if "email" in X:
            X.drop(columns=["email"], inplace=True)
        X = pd.get_dummies(X)
        X = X.reindex(columns=sorted(X.columns), fill_value=0)

        previsoes = {}
        for campo in ["titulacao", "email_ext", "referencia", "statusAtividade"]:
            if campo not in entrada:
                previsoes[campo] = predict_value(campo, X)

        if "nome" in entrada:
            nome = entrada["nome"]
            emails_existentes = load_professores()["email"].tolist()
            previsoes["email"] = gerar_email_unico(nome, previsoes["email_ext"], emails_existentes)
            previsoes["lattes"] = gerar_lattes(nome)

        return previsoes

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição parcial: {str(e)}")
