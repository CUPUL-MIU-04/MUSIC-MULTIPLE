FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar TODO el proyecto
COPY . .

# Crear __init__.py si no existe
RUN touch api/__init__.py && touch audiocraft/__init__.py && touch audiocraft/models/__init__.py

# Exponer puerto
EXPOSE 10000

# Comando para ejecutar
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]
