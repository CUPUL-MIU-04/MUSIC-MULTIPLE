FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema para AV
RUN apt-get update && apt-get install -y \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar AV específicamente (requerido por audiocraft)
RUN pip install --no-cache-dir av

# Copiar TODO el código
COPY . .

# Crear archivos __init__.py necesarios
RUN touch api/__init__.py && touch audiocraft/__init__.py && touch audiocraft/models/__init__.py

# Exponer el puerto
EXPOSE 10000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]
