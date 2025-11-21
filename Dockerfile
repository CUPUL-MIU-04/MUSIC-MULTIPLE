FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
