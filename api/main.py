from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import torch
import torchaudio
from auth import verify_api_key
import os
import sys

# Agregar el path actual para importar audiocraft
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Intentar importar audiocraft
try:
    # Importar desde la estructura correcta
    from audiocraft.models import musicgen
    from audiocraft.data.audio import audio_write
    MODEL_LOADED = True
    print("✅ Audiocraft importado correctamente")
except ImportError as e:
    print(f"❌ Error importando audiocraft: {e}")
    MODEL_LOADED = False

app = FastAPI(title="Music Multiple API", description="API para generación de música desde texto")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class TextToMusicRequest(BaseModel):
    text: str
    duration: int = 10

class MusicResponse(BaseModel):
    success: bool
    message: str
    audio_data: str = None
    duration: float = 0

@app.on_event("startup")
async def startup_event():
    global model
    if not MODEL_LOADED:
        print("🔶 Modo simulación - Audiocraft no disponible")
        return
        
    try:
        print("🎵 Cargando modelo MusicGen...")
        # Cargar MusicGen (modelo más pequeño para ahorrar memoria)
        model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small')
        model.to(device)
        model.set_generation_params(duration=10)
        print("✅ Modelo MusicGen cargado exitosamente")
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        model = None

@app.post("/generate-music")
async def generate_music(request: TextToMusicRequest, authorized: bool = Depends(verify_api_key)):
    try:
        if model is None:
            return await generate_simulated_audio(request.text, request.duration)
        
        print(f"🎶 Generando música para: {request.text}")
        
        with torch.no_grad():
            model.set_generation_params(duration=request.duration)
            generated_audio = model.generate(
                descriptions=[request.text],
                progress=False
            )
        
        # Convertir tensor a audio
        sample_rate = model.sample_rate
        audio_tensor = generated_audio[0].cpu()
        
        # Guardar en buffer WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, sample_rate, format='wav')
        buffer.seek(0)
        
        # Codificar en base64
        audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return MusicResponse(
            success=True,
            message=f"Música generada: {request.text}",
            audio_data=audio_b64,
            duration=request.duration
        )
        
    except Exception as e:
        print(f"❌ Error en generación: {e}")
        return await generate_simulated_audio(request.text, request.duration)

async def generate_simulated_audio(text: str, duration: int):
    """Generar audio simulado como fallback"""
    import numpy as np
    from scipy.io.wavfile import write
    
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Melodía simple basada en el texto
    base_freq = 220 + (hash(text) % 500)
    melody = np.sin(2 * np.pi * base_freq * t) * 0.3
    
    # Convertir a bytes WAV
    buffer = io.BytesIO()
    write(buffer, sample_rate, (melody * 32767).astype(np.int16))
    buffer.seek(0)
    
    audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return MusicResponse(
        success=True,
        message=f"Simulación: {text}",
        audio_data=audio_b64,
        duration=duration
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "device": device,
        "audiocraft_available": MODEL_LOADED
    }

@app.get("/")
async def root():
    return {"message": "Music Multiple API funcionando", "version": "1.0"}

# Mantener el endpoint predict existente para compatibilidad
class AudioFeatures(BaseModel):
    features: list

@app.post("/predict")
async def predict_music(audio_features: AudioFeatures, authorized: bool = Depends(verify_api_key)):
    return {
        "prediction": [0.8, 0.1, 0.1],
        "success": True,
        "message": "Predicción exitosa"
    }
