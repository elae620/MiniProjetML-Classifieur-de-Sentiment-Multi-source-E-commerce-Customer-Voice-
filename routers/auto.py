"""
routers/auto.py — Endpoint de prédiction avec détection automatique du type de modalité.
"""

from fastapi import APIRouter, File, UploadFile, status
from schemas.prediction import UnifiedPredictionResponse
from services.prediction_service import prediction_service

router = APIRouter(prefix="", tags=["Prédiction Automatique"])


@router.post(
    "/predict",
    response_model=UnifiedPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Détection automatique et prédiction unifiée",
    description=(
        "Endpoint polymorphe : reçoit un fichier entrant (image, audio ou texte), "
        "détecte automatiquement sa nature, applique la brique correspondante et retourne la prédiction."
    ),
)
def predict_auto_endpoint(
    file: UploadFile = File(..., description="Fichier média à analyser (image, audio ou texte)")
) -> UnifiedPredictionResponse:
    """
    Endpoint POST /predict (Détection auto du type).
    """
    file_bytes = file.file.read()
    return prediction_service.process_auto(
        file_bytes=file_bytes,
        filename=file.filename or "upload.bin",
        content_type=file.content_type,
    )
