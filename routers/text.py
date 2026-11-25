"""
routers/text.py — Endpoint de prédiction pour les avis au format texte brut.
"""

from fastapi import APIRouter, status
from schemas.prediction import TextRequest, UnifiedPredictionResponse
from services.prediction_service import prediction_service

router = APIRouter(prefix="", tags=["Prédiction Texte"])


@router.post(
    "/predict/text",
    response_model=UnifiedPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prédire le sentiment d'un avis textuel",
    description="Reçoit un corps JSON contenant le texte de l'avis et renvoie la prédiction de sentiment.",
)
def predict_text_endpoint(payload: TextRequest) -> UnifiedPredictionResponse:
    """
    Endpoint POST /predict/text.
    - Reçoit : JSON `{"texte": "..."}`
    - Renvoie : JSON unifié avec `source: "text"`
    """
    return prediction_service.process_text(payload.texte)
