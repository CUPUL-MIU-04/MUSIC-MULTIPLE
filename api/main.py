from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import base64
from io import BytesIO

# Importa tu modelo de generación de música aquí
# from your_music_model import generate_music_from_text

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
    duration: int = 30
    format: str = "wav"

class MusicResponse(BaseModel):
    success: bool
    message: str
    audio_data: str = None  # Base64 encoded audio
    audio_url: str = None

@app.post("/generate-music")
async def generate_music(request: TextToMusicRequest):
    try:
        # Aquí integras tu modelo real
        # audio_data = generate_music_from_text(request.text, request.duration)
        
        # Por ahora, simulamos la respuesta
        # En producción, esto generaría audio real con tu modelo
        
        # Simulación: crear audio simple (reemplaza con tu modelo)
        import numpy as np
        from scipy.io.wavfile import write
        import io
        
        sample_rate = 44100
        duration = request.duration
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generar audio basado en el texto (simulación)
        frequency = 440  # Hz
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Convertir a bytes
        audio_bytes = io.BytesIO()
        write(audio_bytes, sample_rate, audio_data)
        audio_bytes.seek(0)
        
        # Codificar en base64 para enviar por JSON
        audio_b64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        
        return MusicResponse(
            success=True,
            message="Música generada exitosamente",
            audio_data=audio_b64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando música: {str(e)}")
