"""
core/config.py — Configuration centrale de l'application FastAPI.

Centralise les constantes, paramètres de sécurité (tailles maximales, types MIME)
et chemins des artefacts de modèles.
"""

from pathlib import Path
from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    PROJECT_NAME: str = "Classifieur de Sentiment Multi-source"
    PROJECT_VERSION: str = "1.0.0"
    API_PREFIX: str = ""

    # Chemins
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: Path = BASE_DIR / "models" / "model.pkl"
    VECTORIZER_PATH: Path = BASE_DIR / "models" / "vectorizer.pkl"

    # Limitations & Validation de sécurité
    MAX_TEXT_LENGTH: int = 5000         # Caractères max pour un texte
    MAX_IMAGE_SIZE_MB: int = 10         # 10 Mo max pour une image
    MAX_AUDIO_SIZE_MB: int = 25         # 25 Mo max pour un audio

    ALLOWED_IMAGE_MIMES: List[str] = [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/bmp",
        "image/tiff",
    ]

    ALLOWED_AUDIO_MIMES: List[str] = [
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/ogg",
        "audio/webm",
        "audio/x-m4a",
        "audio/flac",
    ]

    # CORS
    CORS_ORIGINS: List[str] = ["*"]


settings = Settings()
