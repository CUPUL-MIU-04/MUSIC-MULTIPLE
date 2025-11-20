FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar TODO primero
COPY . .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r api/requirements.txt

# Exponer el puerto
EXPOSE 10000

# Comando para ejecutar
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]
