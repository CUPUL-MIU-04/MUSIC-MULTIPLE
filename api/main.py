from fastapi import HTTPException, Header
from typing import Annotated

# API Keys válidas
VALID_API_KEYS = [
    "music_ai_key_gerardo_2024",
    "cupul_miu_04_music_key", 
    "test_key_12345"
]

async def verify_api_key(api_key: Annotated[str | None, Header()] = None):
    if not api_key:
        raise HTTPException(status_code=401, detail="API key requerida")
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="API key inválida")
    return True

# El resto de tu código main.py sigue igual...
