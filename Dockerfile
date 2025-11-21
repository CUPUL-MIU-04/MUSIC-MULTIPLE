FROM python:3.11-slim

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar TODO el proyecto (incluyendo audiocraft)
COPY . .

# Exponer el puerto
EXPOSE 10000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]
