FROM python:3.10.0-slim

WORKDIR /app

# Instale dependências do sistema necessárias para mysqlclient e build
RUN apt-get update && \
    apt-get install -y build-essential default-libmysqlclient-dev pkg-config && \
    apt-get clean

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3001

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "3001"]