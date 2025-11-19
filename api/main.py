from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from auth import verify_api_key
import os
import torch
import torchaudio
import librosa
from io import BytesIO

app = FastAPI(title="Music Multiple AI API", description="API para generación de música con IA")

# Configurar CORS para el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, cambia a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resto del código igual...
