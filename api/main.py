from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import base64
import io
import torch
import torchaudio
from auth import verify_api_key
import os
import sys

# Agregar el path actual para importar audiocraft
sys.path.append('/app')

# Configuración de modelo
MODEL_LOADED = False
music_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🎵 Inicializando Music Multiple API...")
print(f"📱 Dispositivo: {device}")

# Intentar importar audiocraft después de instalar soundfile
try:
    import soundfile as sf
    print("✅ soundfile importado exitosamente")
    
    # Ahora intentar importar audiocraft
    from audiocraft.models.music_multiple import MusicMultiple
    from audiocraft.models.musicgen import MusicGen
    MODEL_LOADED = True
    print("✅ Audiocraft importado exitosamente")
    
except ImportError as e:
    print(f"❌ Error importando dependencias: {e}")
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

class TextToMusicRequest(BaseModel):
    text: str
    duration: int = 10

class MusicResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
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
        
        # Intentar cargar modelos en orden de preferencia
        models_to_try = [
            ("MusicMultiple", lambda: MusicMultiple.get_pretrained('facebook/musicgen-melody')),
            ("MusicGen Melody", lambda: MusicGen.get_pretrained('facebook/musicgen-melody')),
            ("MusicGen Small", lambda: MusicGen.get_pretrained('facebook/musicgen-small')),
        ]
        
        for model_name, loader in models_to_try:
            try:
                music_model = loader()
                print(f"✅ {model_name} cargado exitosamente")
                break
            except Exception as e:
                print(f"❌ Error cargando {model_name}: {e}")
                continue
        
        if music_model:
            music_model.to(device)
            music_model.set_generation_params(duration=10)
            print(f"🎉 Modelo listo en {device}!")
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
        
        print(f"🎵 Generando música para: '{request.text}'")
        
        with torch.no_grad():
            music_model.set_generation_params(duration=request.duration)
            generated_audio = music_model.generate(
                descriptions=[request.text],
                progress=True
            )
        
        # Procesar audio
        sample_rate = music_model.sample_rate
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
            model_used="musicgen"
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
    
    # Melodía más interesante basada en el texto
    text_hash = abs(hash(text)) % 1000
    base_freq = 200 + (text_hash % 200)
    
    # Crear melodía con progresión
    melody = np.zeros_like(t)
    for i in range(4):
        start = i * duration / 4
        end = (i + 1) * duration / 4
        mask = (t >= start) & (t < end)
        chord_freq = base_freq * (1 + i * 0.2)
        melody[mask] = np.sin(2 * np.pi * chord_freq * t[mask]) * 0.2
    
    # Agregar bajo
    bass_freq = base_freq * 0.5
    bass = np.sin(2 * np.pi * bass_freq * t) * 0.1
    
    # Agregar ritmo
    beat = np.sin(2 * np.pi * 4 * t) * 0.05
    
    audio_data = (melody + bass + beat) * 0.3
    
    # Convertir a WAV
    buffer = io.BytesIO()
    write(buffer, sample_rate, (audio_data * 32767).astype(np.int16))
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
