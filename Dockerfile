FROM python:3.11-slim

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la API
COPY api/ .

# Exponer el puerto
EXPOSE 10000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
