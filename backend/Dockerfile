
# Usa imagem base com Python
FROM python:3.10-slim

# Define o diretório de trabalho
WORKDIR /app

COPY requirements.txt .

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expõe a porta 8080 (usada pelo Cloud Run)
EXPOSE 8080

# Comando para rodar a aplicação
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]