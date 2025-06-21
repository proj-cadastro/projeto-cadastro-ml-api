# 📚 API de Cadastro Inteligente de Professores

Esta API realiza o cadastro inteligente de professores, prevendo e completando automaticamente campos com base em informações parciais fornecidas pelo usuário. Utiliza modelos de Machine Learning (Árvore de Decisão e Rede Neural) e algoritmos estatísticos para oferecer sugestões confiáveis, realistas e variadas, baseando-se em dados reais de professores.

---

## 🚀 Funcionalidades

- **Predição Completa (`/predict/full`)**: Gera um cadastro completo de professor, prevendo todos os campos relevantes usando os melhores modelos de ML treinados.
- **Predição Parcial (`/predict/partial`)**: Recebe um JSON com informações parciais e retorna os campos faltantes previstos automaticamente, utilizando modelos parciais de ML ou algoritmos estatísticos, conforme o cenário.
- **Re-treinamento Manual (`/train`)**: Permite re-treinar todos os modelos de Machine Learning (completos e parciais) e gerar relatórios de desempenho sob demanda.
- **Re-treinamento Automático**: O sistema verifica periodicamente (a cada 2 horas) se houve alteração significativa no banco de dados e, se necessário, re-treina todos os modelos e atualiza os relatórios.
- **Geração de Relatórios**: Salva métricas detalhadas, matrizes de confusão e comparativos de desempenho dos modelos em `/docs`.
- **Geração Única de E-mail e Lattes**: Cria e-mail e link Lattes exclusivos baseados no nome do professor.
- **Autenticação via API Key**: Todas as rotas são protegidas por chave de API.
- **Verificação Automática de Modelos**: Antes de cada requisição, a API garante que todos os modelos necessários estejam treinados e prontos para uso.

---

## 🧠 Pipeline e Arquitetura

### 1. **Pré-processamento de Dados**
- Os dados são carregados do banco de dados MySQL ou de um arquivo CSV, conforme configuração.
- Realiza-se limpeza, tratamento de valores ausentes e transformação de variáveis categóricas em dummies para uso nos modelos.

### 2. **Treinamento de Modelos**
- **Modelos Completos**: Para cada campo previsto por ML (`titulacao`, `email_ext`, `referencia`, `statusAtividade`), são treinados dois modelos: Árvore de Decisão e Rede Neural.
- **Modelos Parciais**: Para predição parcial, são treinados 6 modelos específicos:
  - 3 modelos para prever cada campo individualmente (quando apenas 1 dos 3 está ausente).
  - 3 modelos para prever pares de campos (quando 2 dos 3 estão ausentes).
- Os modelos são salvos em `modelos_treinados/` (completos) e `modelos_treinados/partial/` (parciais).

### 3. **Geração de Relatórios**
- Para cada campo e tipo de modelo, é gerado um relatório detalhado em `/docs/decision_tree/` e `/docs/neural_network/`, contendo:
  - Observação sobre o uso do modelo (completo ou parcial).
  - Arquitetura do modelo (para redes neurais).
  - Hiperparâmetros.
  - Métricas: acurácia, precisão, recall, F1-score.
  - Matriz de confusão.
  - Análise de overfitting/underfitting.
- Um relatório comparativo (`/docs/comparativo.txt`) resume o desempenho dos modelos e indica o melhor modelo para cada campo.

### 4. **Predição**
- **Predição Completa**: Utiliza sempre o melhor modelo treinado para cada campo, conforme o relatório comparativo.
- **Predição Parcial**:
  - Se o usuário fornecer pelo menos 1 dos 3 campos de ML (`titulacao`, `referencia`, `statusAtividade`), utiliza o modelo parcial correspondente para prever os campos faltantes.
  - Se nenhum desses campos for fornecido, utiliza algoritmos estatísticos que sorteiam entre os 3 valores mais comuns de cada campo, garantindo variedade e evitando respostas "viciadas".
- **Geração de Nome, E-mail e Lattes**: Sempre que necessário, gera valores únicos e consistentes para esses campos, baseando-se em listas de nomes reais e regras de formatação.

### 5. **Verificação e Re-treinamento**
- Antes de cada requisição, a API verifica se todos os modelos necessários estão treinados. Se não estiverem, realiza o treinamento automaticamente.
- O re-treinamento automático é disparado a cada 2 horas, ou sempre que houver alteração significativa no número de registros do banco de dados.

---

## 🔍 Algoritmos Utilizados

### Modelos de Machine Learning

- **Árvore de Decisão (DecisionTreeClassifier)**: Utilizada para classificação dos campos categóricos. Permite interpretar a importância das variáveis e é eficiente para conjuntos de dados menores.
- **Rede Neural (Keras Sequential)**: Utilizada para classificação, especialmente quando há maior complexidade ou não-linearidade nos dados. Arquitetura e hiperparâmetros são definidos e registrados nos relatórios.

### Algoritmo Estatístico para Predição Parcial

- Quando nenhum campo de ML é fornecido, a API sorteia o valor de cada campo entre os 3 mais comuns do banco, garantindo variedade e evitando respostas repetitivas.
- Para o campo `referencia`, o sorteio é feito entre as 3 referências mais frequentes, respeitando a ordem de progressão das referências.

---

## 🛠️ Como Utilizar

### 1. Clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd projeto-cadastro-ml-api
```

### 2. Crie e ative o ambiente virtual

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure o arquivo `.env`

```bash
cp .env-exemplo .env
```

Preencha com:
- Configurações do banco de dados
- Sua `API_KEY`
- Defina `USE_CSV=true` se quiser usar o CSV ao invés do banco

### 5. Treine os modelos (opcional)

```bash
python main.py
```

### 6. Execute a API

```bash
uvicorn src.app:app --reload
```

---

## 📬 Exemplos de Requisição

### Predição Parcial

**POST /predict/partial**

```
Headers:
apikey: <SUA_API_KEY>
```

#### Exemplo 1: Nenhum campo de ML fornecido

```json
{
  "nome": "Maria Meireles"
}
```

**Resposta:**
```json
{
  "nome": "Maria Meireles",
  "email": "maria.meireles@fatec.sp.gov.br",
  "titulacao": "MESTRE",
  "referencia": "PES_II_F",
  "statusAtividade": "ATIVO",
  "lattes": "https://lattes.com.br/mariameireles"
}
```
> Os campos `titulacao`, `referencia` e `statusAtividade` são sorteados entre os 3 mais comuns do banco.

#### Exemplo 2: Um campo de ML fornecido

```json
{
  "nome": "Maria Meireles",
  "titulacao": "DOUTOR"
}
```

**Resposta:**
```json
{
  "nome": "Maria Meireles",
  "email": "maria.meireles@fatec.sp.gov.br",
  "titulacao": "DOUTOR",
  "referencia": "PES_III_A",
  "statusAtividade": "ATIVO",
  "lattes": "https://lattes.com.br/mariameireles"
}
```
> Os campos `referencia` e `statusAtividade` são previstos por ML, considerando a `titulacao` fornecida.

#### Exemplo 3: Dois campos de ML fornecidos

```json
{
  "nome": "Maria Meireles",
  "titulacao": "MESTRE",
  "referencia": "PES_I_A"
}
```

**Resposta:**
```json
{
  "nome": "Maria Meireles",
  "email": "maria.meireles@fatec.sp.gov.br",
  "titulacao": "MESTRE",
  "referencia": "PES_I_A",
  "statusAtividade": "AFASTADO",
  "lattes": "https://lattes.com.br/mariameireles"
}
```
> Apenas `statusAtividade` é previsto por ML, considerando os campos fornecidos.

---

## 📂 Estrutura de Pastas

```
├── src/
│   ├── app.py
│   ├── controllers.py
│   ├── routes.py
│   ├── ...
│   ├── services/
│   │   ├── train_decision_tree.py
│   │   ├── train_nn.py
│   │   ├── train_partial_models.py
│   │   ├── generate_model_reports.py
│   │   └── ...
│   ├── utils/
│   │   ├── statistics.py
│   │   ├── model_check.py
│   │   ├── ...
│   └── ...
├── docs/
│   ├── comparativo.txt
│   ├── confusion_matrix/
│   │   ├── decision_tree/
│   │   └── neural_network/
│   ├── decision_tree/
│   └── neural_network/
├── modelos_treinados/
│   ├── partial/
│   │   ├── parcial_titulacao.pkl
│   │   ├── parcial_referencia.pkl
│   │   ├── parcial_statusAtividade.pkl
│   │   ├── parcial_titulacao_referencia.pkl
│   │   ├── parcial_titulacao_statusAtividade.pkl
│   │   └── parcial_referencia_statusAtividade.pkl
│   ├── titulacao_model.pkl
│   ├── titulacao_nn.h5
│   ├── ...
├── logs/
│   ├── errors.log
│   ├── predictions.log
│   ├── training.log
│   └── used_database.log
├── .env
├── requirements.txt
└── README.md
```

---

## 📊 Relatórios e Métricas

### Relatórios Individuais

- Para cada campo e tipo de modelo, é gerado um relatório em `/docs/decision_tree/` e `/docs/neural_network/`.
- Cada relatório contém:
  - Observação sobre o uso do modelo (completo ou parcial).
  - Arquitetura do modelo (para redes neurais).
  - Hiperparâmetros.
  - Métricas: acurácia, precisão, recall, F1-score.
  - Matriz de confusão.
  - Análise de overfitting/underfitting.

### Relatório Comparativo

- O arquivo `/docs/comparativo.txt` resume o desempenho dos modelos e indica o melhor modelo para cada campo, com base na acurácia.

#### Exemplo de conteúdo do `comparativo.txt`:

```txt
titulacao:
  Melhor modelo: decision_tree
  Acurácia NN: 0.6200
  Acurácia DT: 0.6300

email_ext:
  Melhor modelo: neural_network
  Acurácia NN: 0.9000
  Acurácia DT: 0.9000

referencia:
  Melhor modelo: decision_tree
  Acurácia NN: 0.1900
  Acurácia DT: 0.2100

statusAtividade:
  Melhor modelo: decision_tree
  Acurácia NN: 0.4100
  Acurácia DT: 0.4400
```

---

## 🔒 Segurança

- Todas as rotas são protegidas por API Key, definida no arquivo `.env`.
- O acesso não autorizado é bloqueado automaticamente.

---

## 🔄 Fluxo de Re-treinamento

- O re-treinamento pode ser disparado manualmente via `/train` ou automaticamente a cada 2 horas.
- O sistema verifica se houve alteração significativa no número de registros do banco de dados antes de re-treinar.
- Todos os modelos (completos e parciais) são treinados e os relatórios são atualizados.

---

## 🧑‍💻 Detalhes Técnicos

- **Framework principal:** FastAPI
- **Machine Learning:** scikit-learn (Árvore de Decisão), TensorFlow/Keras (Rede Neural)
- **Banco de dados:** MySQL (ou CSV para testes)
- **Agendamento:** APScheduler
- **Logs:** Todos os eventos relevantes são registrados em arquivos de log na pasta `/logs`.

---

## 📝 Observações Finais

- O sistema foi projetado para ser robusto, flexível e facilmente extensível.
- O uso de modelos parciais garante predições realistas mesmo com informações incompletas.
- O algoritmo estatístico evita respostas repetitivas e "viciadas", tornando a API mais próxima de um sistema real de apoio à decisão.
- Toda a lógica de verificação e re-treinamento é automática, garantindo que a API esteja sempre pronta para uso.

---