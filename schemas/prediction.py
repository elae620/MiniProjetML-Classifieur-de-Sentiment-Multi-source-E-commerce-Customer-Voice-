"""
schemas/prediction.py — Schémas Pydantic de validation des requêtes et réponses.

Respecte strictement le format unifié spécifié dans le cahier des charges :
{
  "source": "text|image|audio",
  "texte_extrait": "...",
  "sentiment_predit": "...",
  "confiance": 0.89
}
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class TextRequest(BaseModel):
    """Corps de requête pour la prédiction textuelle (POST /predict/text)."""
    texte: str = Field(
        ...,
        description="Avis client en texte brut",
        examples=["Produit conforme à la description, livraison rapide et soignée !"]
    )

    @field_validator("texte")
    @classmethod
    def valider_texte_non_vide(cls, v: str) -> str:
        nettoye = (v or "").strip()
        if not nettoye:
            raise ValueError("Le champ 'texte' ne peut pas être vide ou constitué uniquement d'espaces.")
        if len(nettoye) > 5000:
            raise ValueError("Le texte dépasse la taille maximale autorisée (5000 caractères).")
        return nettoye


class UnifiedPredictionResponse(BaseModel):
    """
    Format JSON de réponse unifié pour TOUS les endpoints de prédiction
    (spécification exacte du sujet d'examen).
    """
    source: Literal["text", "image", "audio"] = Field(
        ...,
        description="Modalité d'entrée analysée ('text', 'image' ou 'audio')",
        examples=["text"]
    )
    texte_extrait: str = Field(
        ...,
        description="Texte analysé (saisi directement ou extrait via OCR/ASR)",
        examples=["Produit conforme à la description, livraison rapide et soignée !"]
    )
    sentiment_predit: Literal["positif", "negatif", "neutre"] = Field(
        ...,
        description="Sentiment prédit par le modèle de classification",
        examples=["positif"]
    )
    confiance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Indice de confiance du modèle sur la prédiction (entre 0.0 et 1.0)",
        examples=[0.89]
    )


class HealthResponse(BaseModel):
    """Réponse du point de terminaison de santé GET /health."""
    status: str = Field(default="ok", examples=["ok"])
    model_loaded: bool = Field(default=True, examples=[True])
    version: str = Field(default="1.0.0", examples=["1.0.0"])
