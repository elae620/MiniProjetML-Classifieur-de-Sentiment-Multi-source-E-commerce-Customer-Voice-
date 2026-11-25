"""
routers/audio.py — Endpoint de prédiction pour les fichiers audio (ASR -> Classifieur).
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from core.config import settings
from schemas.prediction import UnifiedPredictionResponse
from services.prediction_service import prediction_service

router = APIRouter(prefix="", tags=["Prédiction Audio (ASR)"])


@router.post(
    "/predict/audio",
    response_model=UnifiedPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prédire le sentiment depuis un fichier audio",
    description=(
        "Reçoit un fichier audio vocal (multipart/form-data), transcrit la voix par ASR (Whisper) "
        "et prédit le sentiment avec le modèle textuel."
    ),
)
def predict_audio_endpoint(
    file: UploadFile = File(..., description="Fichier audio vocal (WAV, MP3, etc.)")
) -> UnifiedPredictionResponse:
    """
    Endpoint POST /predict/audio.
    - Reçoit : Fichier audio multipart/form-data
    - Renvoie : JSON unifié avec `source: "audio"`
    """
    # 1. Validation type MIME / extension
    content_type = (file.content_type or "").lower()
    ext = ("." + file.filename.split(".")[-1].lower()) if file.filename and "." in file.filename else ""

    if (
        content_type not in settings.ALLOWED_AUDIO_MIMES
        and not content_type.startswith("audio/")
        and ext not in {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"}
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Format de fichier non autorisé : '{content_type}'. "
                f"Seuls les fichiers audio sont acceptés (WAV, MP3, etc.)."
            ),
        )

    # 2. Lecture synchrone du contenu
    audio_bytes = file.file.read()

    # 3. Traitement
    return prediction_service.process_audio(
        audio_bytes=audio_bytes,
        filename=file.filename or "upload.mp3"
    )
