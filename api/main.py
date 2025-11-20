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
import typing as tp

# Agregar el path actual para importar audiocraft
sys.path.append('/app')

# Intentar importar tu modelo personalizado
try:
    # Importar directamente desde la estructura de archivos
    from audiocraft.models.musicgen import MusicGen
    from audiocraft.models.music_multiple import MusicMultiple
    MODEL_LOADED = True
    print("✅ Módulos de audiocraft importados exitosamente")
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
    duration: int = 30
    genre: tp.Optional[str] = None
    subgenre: tp.Optional[str] = None
    duration_type: tp.Optional[str] = "medium"

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
        print("🎵 Cargando modelo Music Multiple...")
        
        # Cargar usando tu método get_pretrained personalizado
        music_model = MusicMultiple.get_pretrained('facebook/musicgen-small', device=device)
        print("✅ MusicMultiple cargado exitosamente")
        
        # Configurar duración por defecto
        if hasattr(music_model, 'set_duration'):
            music_model.set_duration('medium')
        else:
            music_model.set_generation_params(duration=30)
        
        print("🎉 Modelo Music Multiple listo para generar música!")
        
        # Mostrar géneros disponibles
        if hasattr(music_model, 'list_available_genres'):
            music_model.list_available_genres()
            
    except Exception as e:
        print(f"❌ Error cargando MusicMultiple: {e}")
        music_model = None

@app.post("/generate-music")
async def generate_music(request: TextToMusicRequest, authorized: bool = Depends(verify_api_key)):
    try:
        if music_model is None or not MODEL_LOADED:
            # Modo simulación
            return await generate_simulated_audio(request.text, request.duration)
        
        print(f"🎵 Generando música para: '{request.text}'")
        
        with torch.no_grad():
            # Configurar duración
            if hasattr(music_model, 'set_duration') and request.duration_type:
                try:
                    music_model.set_duration(request.duration_type)
                except ValueError as e:
                    print(f"⚠️  Error configurando duración: {e}")
                    music_model.set_generation_params(duration=request.duration)
            else:
                music_model.set_generation_params(duration=request.duration)
            
            # Generar música según los parámetros
            if request.genre:
                # Usar generación con género específico
                print(f"🎶 Usando género: {request.genre}, subgénero: {request.subgenre}")
                generated_audio = music_model.generate_with_genre(
                    descriptions=[request.text],
                    genre=request.genre,
                    subgenre=request.subgenre,
                    progress=False
                )
            else:
                # Generación normal
                generated_audio = music_model.generate(
                    descriptions=[request.text],
                    progress=False
                )
        
        # Procesar el audio generado
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
    
    # Melodía más interesante para la simulación
    base_freq = 220 + (hash(text) % 300)
    melody = np.zeros_like(t)
    
    # Crear una progresión de acordes simple
    chords = [base_freq, base_freq * 1.25, base_freq * 1.5]
    for i, freq in enumerate(chords):
        start = i * duration / len(chords)
        end = (i + 1) * duration / len(chords)
        mask = (t >= start) & (t < end)
        melody[mask] = np.sin(2 * np.pi * freq * t[mask]) * 0.2
    
    # Agregar ritmo
    beat_freq = 2  # Hz
    rhythm = np.sin(2 * np.pi * beat_freq * t) * 0.1
    melody += rhythm
    
    # Normalizar
    melody = melody * 0.3
    
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

# Endpoints adicionales para explorar capacidades del modelo
@app.get("/genres")
async def list_genres():
    """Listar todos los géneros y subgéneros disponibles"""
    if music_model is None or not hasattr(music_model, 'available_genres'):
        return {"error": "Modelo no disponible o no tiene géneros configurados"}
    
    return {
        "success": True,
        "genres": music_model.available_genres
    }

@app.get("/durations")
async def list_durations():
    """Listar todas las duraciones preconfiguradas"""
    if music_model is None or not hasattr(music_model, 'available_durations'):
        return {"error": "Modelo no disponible o no tiene duraciones configuradas"}
    
    return {
        "success": True, 
        "durations": music_model.available_durations
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": music_model is not None,
        "audiocraft_available": MODEL_LOADED,
        "device": device,
        "model_type": "MusicMultiple" if music_model else "None"
    }

@app.get("/")
async def root():
    return {
        "message": "Music Multiple API funcionando", 
        "version": "1.0",
        "features": [
            "Generación de música desde texto",
            "Soporte para géneros latinos",
            "Procesamiento en español", 
            "Múltiples duraciones preconfiguradas"
        ]
    }
