"""
services/prediction_service.py — Orchestration métier de l'architecture en cascade.

Rassemble les briques :
  - Text : modules.classifier
  - Image : modules.ocr -> modules.classifier
  - Audio : modules.asr -> modules.classifier
  - Auto : détection automatique de modalité selon MIME/extension -> cascade
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from fastapi import HTTPException, status

from core.config import settings
from modules.classifier import predict
from modules.ocr import extraire_texte_image
from modules.asr import transcrire_audio
from schemas.prediction import UnifiedPredictionResponse

log = logging.getLogger(__name__)


class PredictionService:
    """Service d'orchestration de la classification en cascade."""

    @staticmethod
    def process_text(texte: str) -> UnifiedPredictionResponse:
        """
        Traite une requête textuelle directe.
        """
        texte_nettoye = (texte or "").strip()
        if not texte_nettoye:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Le texte fourni est vide après nettoyage.",
            )

        try:
            sentiment, confiance = predict(texte_nettoye)
        except Exception as exc:
            log.error("Erreur classification texte : %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erreur interne lors de la prédiction : {exc}",
            )

        return UnifiedPredictionResponse(
            source="text",
            texte_extrait=texte_nettoye,
            sentiment_predit=sentiment,
            confiance=confiance,
        )

    @staticmethod
    def process_image(image_bytes: bytes, filename: str = "upload.png") -> UnifiedPredictionResponse:
        """
        Traite une requête image : OCR (Pillow + Tesseract) -> Classifieur.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Le fichier image envoyé est vide (0 octet).",
            )

        # Taille maximale
        max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
        if len(image_bytes) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"L'image dépasse la taille maximale autorisée ({settings.MAX_IMAGE_SIZE_MB} Mo).",
            )

        # 1. Extraction OCR
        try:
            texte_extrait = extraire_texte_image(image_bytes)
        except RuntimeError as e:
            # Tesseract manquant ou erreur système OCR
            log.error("Erreur système OCR : %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Module OCR indisponible : {e}",
            )
        except Exception as e:
            log.error("Erreur traitement image OCR : %s", e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Échec de l'extraction OCR sur l'image : {e}",
            )

        if not texte_extrait or not texte_extrait.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Aucun texte lisible n'a pu être extrait de l'image fournie.",
            )

        # 2. Classification
        sentiment, confiance = predict(texte_extrait)

        return UnifiedPredictionResponse(
            source="image",
            texte_extrait=texte_extrait,
            sentiment_predit=sentiment,
            confiance=confiance,
        )

    @staticmethod
    def process_audio(audio_bytes: bytes, filename: str = "upload.mp3") -> UnifiedPredictionResponse:
        """
        Traite une requête audio : ASR (Whisper) -> Classifieur.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Le fichier audio envoyé est vide (0 octet).",
            )

        # Taille maximale
        max_bytes = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024
        if len(audio_bytes) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"L'audio dépasse la taille maximale autorisée ({settings.MAX_AUDIO_SIZE_MB} Mo).",
            )

        # 1. Transcription ASR
        try:
            texte_extrait = transcrire_audio(audio_bytes)
        except RuntimeError as e:
            log.error("Erreur transcription ASR : %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Module ASR indisponible ou erreur ffmpeg : {e}",
            )
        except Exception as e:
            log.error("Erreur transcription ASR : %s", e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Échec de la transcription audio : {e}",
            )

        if not texte_extrait or not texte_extrait.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Aucune parole intelligible n'a pu être transcrite depuis le fichier audio.",
            )

        # 2. Classification
        sentiment, confiance = predict(texte_extrait)

        return UnifiedPredictionResponse(
            source="audio",
            texte_extrait=texte_extrait,
            sentiment_predit=sentiment,
            confiance=confiance,
        )

    @classmethod
    def process_auto(
        cls,
        file_bytes: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> UnifiedPredictionResponse:
        """
        Détecte automatiquement si le fichier est une image ou un audio
        puis redirige vers le pipeline correspondant.
        """
        ext = Path(filename).suffix.lower() if filename else ""
        mime = (content_type or "").lower()

        # Détection Image
        if (
            mime.startswith("image/")
            or ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
            or mime in settings.ALLOWED_IMAGE_MIMES
        ):
            return cls.process_image(file_bytes, filename=filename)

        # Détection Audio
        if (
            mime.startswith("audio/")
            or ext in {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"}
            or mime in settings.ALLOWED_AUDIO_MIMES
        ):
            return cls.process_audio(file_bytes, filename=filename)

        # Si le fichier est du texte brut
        if mime.startswith("text/") or ext in {".txt", ".csv"}:
            try:
                text_content = file_bytes.decode("utf-8")
                return cls.process_text(text_content)
            except UnicodeDecodeError:
                pass

        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Type de fichier non supporté pour la détection automatique : "
                f"nom='{filename}', content_type='{content_type}'. "
                f"Formats acceptés : images (PNG, JPEG, etc.) ou audios (WAV, MP3, etc.)."
            ),
        )


prediction_service = PredictionService()
