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

# Agregar el path a tu repositorio para importar los modelos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Intentar importar tu modelo personalizado
try:
    from audiocraft.models import MusicMultiple, musicgen
    from audiocraft.data.audio import audio_write
    MODEL_LOADED = True
except ImportError as e:
    print(f"Error importando modelos: {e}")
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

# Modelo global
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class TextToMusicRequest(BaseModel):
    text: str
    duration: int = 10
    format: str = "wav"

class MusicResponse(BaseModel):
    success: bool
    message: str
    audio_data: str = None  # Base64 encoded audio
    duration: float = 0

@app.on_event("startup")
async def startup_event():
    global model
    if not MODEL_LOADED:
        print("Modelos no disponibles - modo simulación")
        return
        
    try:
        # Intentar cargar tu modelo personalizado MusicMultiple
        # O usar MusicGen si tu modelo no está disponible
        print("Cargando modelo de música...")
        
        # Opción 1: Cargar MusicMultiple (tu modelo personalizado)
        try:
            model = MusicMultiple.get_pretrained('your-model-name')
            print("✅ MusicMultiple cargado exitosamente")
        except:
            # Opción 2: Cargar MusicGen estándar como fallback
            model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small')
            print("✅ MusicGen estándar cargado como fallback")
        
        model.to(device)
        model.set_generation_params(duration=10)  # 10 segundos por defecto
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        model = None

@app.post("/generate-music")
async def generate_music(request: TextToMusicRequest, authorized: bool = Depends(verify_api_key)):
    try:
        if model is None:
            # Modo simulación si el modelo no está cargado
            return await generate_simulated_audio(request.text, request.duration)
        
        # Generar música con el modelo real
        print(f"Generando música para: {request.text}")
        
        with torch.no_grad():
            # Generar audio desde el texto
            model.set_generation_params(duration=request.duration)
            generated_audio = model.generate(
                descriptions=[request.text],
                progress=True
            )
        
        # Convertir tensor a audio
        # generated_audio es de forma (1, 1, sample_rate * duration)
        sample_rate = model.sample_rate
        audio_tensor = generated_audio[0]  # Primera muestra
        
        # Guardar en buffer WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor.cpu(), sample_rate, format='wav')
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
        print(f"Error en generación: {e}")
        # Fallback a simulación
        return await generate_simulated_audio(request.text, request.duration)

async def generate_simulated_audio(text: str, duration: int):
    """Generar audio simulado como fallback"""
    import numpy as np
    from scipy.io.wavfile import write
    import io
    
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Crear melodía simple basada en el texto
    base_freq = 220 + (hash(text) % 500)  # Frecuencia basada en el texto
    melody = np.sin(2 * np.pi * base_freq * t) * 0.3
    
    # Agregar armónicos
    for harmonic in [2, 3, 4]:
        freq = base_freq * harmonic
        volume = 0.1 / harmonic
        melody += np.sin(2 * np.pi * freq * t) * volume
    
    # Normalizar
    melody = melody * 0.3
    
    # Convertir a bytes WAV
    buffer = io.BytesIO()
    write(buffer, sample_rate, (melody * 32767).astype(np.int16))
    buffer.seek(0)
    
    audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return MusicResponse(
        success=True,
        message=f"Simulación: {text} (modelo no disponible)",
        audio_data=audio_b64,
        duration=duration
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "device": device
    }

@app.get("/")
async def root():
    return {"message": "Music Multiple API funcionando", "version": "1.0"}
