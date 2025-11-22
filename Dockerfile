FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema para audio
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias
COPY requirements.txt .
COPY Music-Model/requirements.txt ./Music-Model-requirements.txt

RUN pip install --no-cache-dir -r requirements.txt -r Music-Model-requirements.txt

# Copiar código
COPY api/ .
COPY Music-Model/ ./Music-Model/

# Crear directorio para modelos
RUN mkdir -p Music-Model/models

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
