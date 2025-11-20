from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import torch
import torchaudio
import os
import sys

# Agregar el path para imports relativos
sys.path.append(os.path.dirname(__file__))

# Importación corregida - prueba diferentes opciones:
try:
    # Opción 1: Importación relativa
    from .auth import verify_api_key
except ImportError:
    try:
        # Opción 2: Importación directa
        from auth import verify_api_key
    except ImportError:
        # Opción 3: Crear función simple si todo falla
        from fastapi import Header
        from typing import Annotated
        
        VALID_API_KEYS = ["cupul_miu_04_music_key"]
        
        async def verify_api_key(api_key: Annotated[str | None, Header()] = None):
            if not api_key or api_key not in VALID_API_KEYS:
                raise HTTPException(status_code=401, detail="API key inválida")
            return True

# El resto de tu código para audiocraft...
try:
    from audiocraft.models.music_multiple import MusicMultiple
    from audiocraft.models.musicgen import MusicGen
    MODEL_LOADED = True
    print("✅ Audiocraft importado exitosamente")
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

# Modelo global
music_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

class TextToMusicRequest(BaseModel):
    text: str
    duration: int = 10

class MusicResponse(BaseModel):
    success: bool
    message: str
    audio_data: str = None
    duration: float = 0
    model_used: str = "simulation"

@app.on_event("startup")
async def startup_event():
    global music_model
    if not MODEL_LOADED:
        print("🔶 Modo simulación - audiocraft no disponible")
        return
        
    try:
        print("🎵 Cargando modelo de música...")
        
        # Opción 1: Intentar cargar MusicMultiple
        try:
            music_model = MusicMultiple.get_pretrained('facebook/musicgen-small')
            print("✅ MusicGen cargado (como MusicMultiple)")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            music_model = None
        
        if music_model:
            music_model.to(device)
            music_model.set_generation_params(duration=10)
            print("🎉 Modelo de música listo!")
        else:
            print("🔶 No se pudo cargar ningún modelo")
            
    except Exception as e:
        print(f"❌ Error en startup: {e}")
        music_model = None

@app.post("/generate-music")
async def generate_music(request: TextToMusicRequest, authorized: bool = Depends(verify_api_key)):
    try:
        if music_model is None or not MODEL_LOADED:
            return await generate_simulated_audio(request.text, request.duration)
        
        print(f"🎵 Generando música para: {request.text}")
        
        with torch.no_grad():
            music_model.set_generation_params(duration=request.duration)
            generated_audio = music_model.generate(
                descriptions=[request.text],
                progress=False
            )
        
        sample_rate = 32000
        audio_tensor = generated_audio[0].cpu()
        
        # Guardar como WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, sample_rate, format='wav')
        buffer.seek(0)
        
        audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return MusicResponse(
            success=True,
            message=f"Música generada: {request.text}",
            audio_data=audio_b64,
            duration=request.duration,
            model_used="music_multiple"
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
    
    # Melodía simple
    base_freq = 220 + (hash(text) % 300)
    melody = np.sin(2 * np.pi * base_freq * t) * 0.3
    
    # Convertir a WAV
    buffer = io.BytesIO()
    write(buffer, sample_rate, (melody * 32767).astype(np.int16))
    buffer.seek(0)
    
    audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return MusicResponse(
        success=True,
        message=f"Simulación: {text}",
        audio_data=audio_b64,
        duration=duration,
        model_used="simulation"
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": music_model is not None,
        "audiocraft_available": MODEL_LOADED,
        "device": device
    }

@app.get("/")
async def root():
    return {"message": "Music Multiple API funcionando", "version": "1.0"}
