from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# IMPORTACIÓN ABSOLUTA CORRECTA:
from auth import verify_api_key  # ← SIN el punto

app = FastAPI(title="Music AI API", description="API para modelo de música con IA")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioFeatures(BaseModel):
    features: list
    audio_data: list = None

class PredictionResponse(BaseModel):
    prediction: list
    success: bool
    message: str

@app.get("/")
async def root():
    return {"message": "Music AI API funcionando", "version": "1.0"}

@app.post("/predict")
async def predict_music(
    audio_features: AudioFeatures, 
    authorized: bool = Depends(verify_api_key)
):
    try:
        # Ejemplo simulado
        mock_prediction = [0.8, 0.1, 0.1]
        return PredictionResponse(
            prediction=mock_prediction,
            success=True,
            message="Predicción exitosa"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": False}

# Configuración para Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
