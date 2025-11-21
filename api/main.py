from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import os
import sys

# Agregar el directorio actual al path para importar audiocraft
sys.path.append('/app')

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
    format: str = "wav"

class MusicResponse(BaseModel):
    success: bool
    message: str
    audio_data: str = None
    duration: float = 0

@app.get("/")
async def root():
    return {"message": "Music Multiple API funcionando", "version": "1.0"}

@app.get("/health")
async def health_check():
    # Verificar si los modelos están disponibles
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Intentar importar audiocraft
        try:
            from audiocraft.models import musicgen
            model_loaded = True
        except ImportError:
            model_loaded = False
            
        return {
            "status": "healthy", 
            "model_loaded": model_loaded,
            "device": device,
            "python_path": sys.path
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/generate-music")
async def generate_music(request: TextToMusicRequest):
    try:
        # Primero, intentar usar el modelo real
        try:
            from audiocraft.models import musicgen
            import torch
            
            # Cargar modelo (hacerlo una vez y cachearlo sería mejor)
            model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small')
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            # Generar música
            model.set_generation_params(duration=request.duration)
            generated_audio = model.generate(descriptions=[request.text])
            
            # Convertir a WAV
            sample_rate = model.sample_rate
            audio_tensor = generated_audio[0].cpu()
            
            # Guardar en buffer
            import torchaudio
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_tensor, sample_rate, format='wav')
            buffer.seek(0)
            
            audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            return MusicResponse(
                success=True,
                message=f"Música generada: {request.text}",
                audio_data=audio_b64,
                duration=request.duration
            )
            
        except ImportError as e:
            # Fallback a simulación si audiocraft no está disponible
            return await generate_simulated_audio(request.text, request.duration)
            
    except Exception as e:
        return await generate_simulated_audio(request.text, request.duration)

async def generate_simulated_audio(text: str, duration: int):
    """Generar audio simulado como fallback"""
    import numpy as np
    from scipy.io.wavfile import write
    import io
    
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Crear melodía simple basada en el texto
    base_freq = 220 + (hash(text) % 500)
    melody = np.sin(2 * np.pi * base_freq * t) * 0.3
    
    # Agregar armónicos
    for harmonic in [2, 3, 4]:
        freq = base_freq * harmonic
        volume = 0.1 / harmonic
        melody += np.sin(2 * np.pi * freq * t) * volume
    
    melody = melody * 0.3
    
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
