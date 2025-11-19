from fastapi import HTTPException, Header
from typing import Annotated

# API Keys válidas - CAMBIA ESTAS KEYS EN PRODUCCIÓN!
VALID_API_KEYS = [
    "music_ai_key_gerardo_2024",
    "cupul_miu_04_music_key",
    "test_key_12345"
]

async def verify_api_key(api_key: Annotated[str | None, Header()] = None):
    """
    Verifica si la API key proporcionada es válida
    Uso: incluir header 'api-key: tu_key_aqui'
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key requerida. Usa el header 'api-key'"
        )
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="API key inválida o expirada"
        )
    
    return True

# Función para generar nuevas API keys (opcional)
def generate_api_key(user_id: str) -> str:
    import hashlib
    import secrets
    salt = secrets.token_hex(16)
    key = hashlib.sha256(f"{user_id}{salt}".encode()).hexdigest()[:32]
    return f"music_ai_{key}"
