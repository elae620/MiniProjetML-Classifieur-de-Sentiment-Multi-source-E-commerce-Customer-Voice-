"""
routers/image.py — Endpoint de prédiction pour les images (OCR -> Classifieur).
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from core.config import settings
from schemas.prediction import UnifiedPredictionResponse
from services.prediction_service import prediction_service

router = APIRouter(prefix="", tags=["Prédiction Image (OCR)"])


@router.post(
    "/predict/image",
    response_model=UnifiedPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prédire le sentiment depuis une image",
    description=(
        "Reçoit un fichier image (multipart/form-data), extrait le texte par OCR (pytesseract) "
        "et prédit le sentiment avec le modèle textuel."
    ),
)
def predict_image_endpoint(
    file: UploadFile = File(..., description="Image à analyser (PNG, JPG, etc.)")
) -> UnifiedPredictionResponse:
    """
    Endpoint POST /predict/image.
    - Reçoit : Fichier image multipart/form-data
    - Renvoie : JSON unifié avec `source: "image"`
    """
    # 1. Validation type MIME
    content_type = (file.content_type or "").lower()
    ext = ("." + file.filename.split(".")[-1].lower()) if file.filename and "." in file.filename else ""

    if content_type not in settings.ALLOWED_IMAGE_MIMES and ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Format de fichier non autorisé : '{content_type}'. "
                f"Seules les images sont acceptées (PNG, JPEG, etc.)."
            ),
        )

    # 2. Lecture synchrone du contenu
    image_bytes = file.file.read()

    # 3. Traitement
    return prediction_service.process_image(
        image_bytes=image_bytes,
        filename=file.filename or "upload.png"
    )
