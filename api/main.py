from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated, List, Optional
import os
import numpy as np
import librosa
import io
import soundfile as sf

# Importar el cargador de modelos
import sys
sys.path.append('..')
from Music-Model.model_loader import music_model

# API Keys desde variables de entorno
api_keys_str = os.environ.get("API_KEYS", "music_ai_key_gerardo_2024,cupul_miu_04_music_key,test_key_12345")
VALID_API_KEYS = [key.strip() for key in api_keys_str.split(",")]

app = FastAPI(title="Music AI API", description="API para clasificación de géneros musicales con IA")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Intentar cargar el modelo al iniciar
try:
    model_loaded = music_model.load_latest_model('../Music-Model/models')
    MODEL_STATUS = "loaded" if model_loaded else "not_loaded"
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    MODEL_STATUS = "error"

async def verify_api_key(api_key: Annotated[str | None, Header()] = None):
    if not api_key:
        raise HTTPException(status_code=401, detail="API key requerida. Usa el header 'api-key'")
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="API key inválida o expirada")
    
    return True

class AudioFeatures(BaseModel):
    features: List[float]
    audio_data: Optional[List[float]] = None

class PredictionResponse(BaseModel):
    prediction: dict
    success: bool
    message: str
    model_status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_status: str
    genres_available: List[str] = []

def extract_audio_features(audio_bytes: bytes) -> List[float]:
    """Extraer características de audio desde bytes"""
    try:
        # Convertir bytes a array de audio
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), duration=30)
        
        features = []
        
        # MFCCs
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        features.extend(mfcc_mean)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sr), axis=1)
        
        features.append(spectral_centroid)
        features.append(spectral_rolloff)
        features.extend(spectral_contrast)
        
        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        features.append(tempo)
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        features.append(zcr)
        
        return features
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando audio: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Music Genre Classification API", 
        "version": "2.0",
        "model_status": MODEL_STATUS
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_music(
    audio_features: AudioFeatures, 
    authorized: bool = Depends(verify_api_key)
):
    try:
        if MODEL_STATUS != "loaded":
            return PredictionResponse(
                prediction={},
                success=False,
                message="Modelo no cargado",
                model_status=MODEL_STATUS
            )
        
        # Predecir usando el modelo cargado
        prediction = music_model.predict_features(audio_features.features)
        
        return PredictionResponse(
            prediction=prediction,
            success=True,
            message="Predicción exitosa",
            model_status=MODEL_STATUS
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict-audio")
async def predict_audio_file(
    file: UploadFile = File(...),
    authorized: bool = Depends(verify_api_key)
):
    try:
        if MODEL_STATUS != "loaded":
            raise HTTPException(status_code=503, detail="Modelo no disponible")
        
        # Leer archivo de audio
        audio_bytes = await file.read()
        
        # Extraer características
        features = extract_audio_features(audio_bytes)
        
        # Predecir
        prediction = music_model.predict_features(features)
        
        return {
            "prediction": prediction,
            "success": True,
            "message": f"Archivo {file.filename} procesado exitosamente",
            "features_used": len(features)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando audio: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    genres = music_model.genre_labels if MODEL_STATUS == "loaded" else []
    
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL_STATUS == "loaded",
        model_status=MODEL_STATUS,
        genres_available=genres
    )

@app.get("/model-info")
async def model_info():
    if MODEL_STATUS != "loaded":
        return {"model_loaded": False, "model_status": MODEL_STATUS}
    
    return {
        "model_loaded": True,
        "model_path": music_model.loaded_model_path,
        "genres_available": music_model.genre_labels,
        "input_dim": music_model.model.input_shape[1],
        "num_classes": len(music_model.genre_labels)
    }

# Configuración para Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
