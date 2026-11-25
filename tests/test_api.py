"""
tests/test_api.py — Suite complète de tests automatisés (Unitaires + Intégration).

Exécution :
    pytest tests/test_api.py -v --cov=.

Couverture complète :
- Modules ML : classification, singleton, validation, normalisation
- Preprocessing : NLP, stop-words avec conservation des négations, nettoyage
- Schémas Pydantic : validation des entrées, des bornes, des types MIME et des formats
- Services : orchestration en cascade, limites de taille, gestion d'erreurs (413, 415, 422, 500)
- Modules multimodaux avec Mocks : OCR (Pillow + Tesseract mocké), ASR (Whisper mocké)
- Health check & configuration
"""

import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException

# ── Imports applicatifs ───────────────────────────────────────────────────────
from modules.classifier import predict, _charger_modele
from schemas.prediction import TextRequest, UnifiedPredictionResponse, HealthResponse
from services.prediction_service import prediction_service
from core.config import settings
from train import nettoyer_texte
import modules.ocr as ocr_mod
import modules.asr as asr_mod


# ═══════════════════════════════════════════════════════════════════════════════
#  1. TESTS DU CLASSIFIEUR TEXTE (modules/classifier.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifier:
    """Tests du classifieur textuel scikit-learn."""

    def test_predict_positif(self):
        label, conf = predict("Produit excellent, je suis ravi de ma commande, livraison rapide !")
        assert label in ("positif", "negatif", "neutre")
        assert 0.0 <= conf <= 1.0

    def test_predict_negatif(self):
        label, conf = predict("Article cassé, service client ne répond jamais, très déçu.")
        assert label in ("positif", "negatif", "neutre")
        assert 0.0 <= conf <= 1.0

    def test_predict_neutre(self):
        label, conf = predict("Commande reçue dans les délais, conforme à la description.")
        assert label in ("positif", "negatif", "neutre")
        assert 0.0 <= conf <= 1.0

    def test_predict_texte_vide_leve_erreur(self):
        with pytest.raises(ValueError, match="vide"):
            predict("")

    def test_predict_espaces_leve_erreur(self):
        with pytest.raises(ValueError, match="vide"):
            predict("     ")

    def test_predict_retourne_tuple(self):
        result = predict("Test de prédiction quelconque")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_predict_classes_valides(self):
        label, _ = predict("Vérification des classes")
        assert label in {"positif", "negatif", "neutre"}

    def test_predict_confiance_bornee(self):
        _, conf = predict("Vérification des bornes de confiance")
        assert 0.0 <= conf <= 1.0

    def test_modele_introuvable_leve_erreur(self):
        with pytest.raises(FileNotFoundError):
            _charger_modele(chemin="fichier_inexistant.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. TESTS DU NETTOYAGE TEXTUEL (train.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNettoyageTexte:
    """Tests de la fonction NLP nettoyer_texte."""

    def test_minuscules(self):
        assert nettoyer_texte("PRODUIT EXCELLENT") == "produit excellent"

    def test_suppression_urls(self):
        result = nettoyer_texte("Voir https://example.com ou www.site.fr pour avis")
        assert "https" not in result
        assert "www" not in result

    def test_conservation_negations(self):
        result = nettoyer_texte("je ne suis pas du tout satisfait")
        assert "ne" in result
        assert "pas" in result

    def test_texte_vide(self):
        assert nettoyer_texte("") == ""

    def test_texte_none(self):
        assert nettoyer_texte(None) == ""

    def test_suppression_chiffres_isoles(self):
        result = nettoyer_texte("livré en 24 heures")
        assert "24" not in result.split()


# ═══════════════════════════════════════════════════════════════════════════════
#  3. TESTS DES SCHÉMAS PYDANTIC (schemas/prediction.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemas:
    """Tests de conformité Pydantic v2."""

    def test_text_request_valide(self):
        req = TextRequest(texte="Avis valide")
        assert req.texte == "Avis valide"

    def test_text_request_vide_rejete(self):
        with pytest.raises(Exception):
            TextRequest(texte="   ")

    def test_text_request_trop_long_rejete(self):
        with pytest.raises(Exception):
            TextRequest(texte="a" * 5001)

    def test_unified_response_structure(self):
        resp = UnifiedPredictionResponse(
            source="text",
            texte_extrait="Exemple",
            sentiment_predit="positif",
            confiance=0.92,
        )
        assert resp.source == "text"
        assert resp.sentiment_predit == "positif"
        assert resp.confiance == 0.92

    def test_unified_response_source_invalide(self):
        with pytest.raises(Exception):
            UnifiedPredictionResponse(
                source="video",  # non supporté
                texte_extrait="Exemple",
                sentiment_predit="positif",
                confiance=0.5,
            )

    def test_health_response(self):
        h = HealthResponse(status="ok", model_loaded=True, version="1.0.0")
        assert h.status == "ok"
        assert h.model_loaded is True


# ═══════════════════════════════════════════════════════════════════════════════
#  4. TESTS DU SERVICE D'ORCHESTRATION (services/prediction_service.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictionService:
    """Tests métier de la cascade."""

    def test_process_text_nominal(self):
        resp = prediction_service.process_text("Superbe produit, livraison impeccable !")
        assert resp.source == "text"
        assert resp.sentiment_predit in {"positif", "negatif", "neutre"}
        assert 0.0 <= resp.confiance <= 1.0

    def test_process_text_vide_422(self):
        with pytest.raises(HTTPException) as exc:
            prediction_service.process_text("   ")
        assert exc.value.status_code == 422

    def test_process_image_vide_422(self):
        with pytest.raises(HTTPException) as exc:
            prediction_service.process_image(b"", filename="test.png")
        assert exc.value.status_code == 422

    def test_process_image_trop_grande_413(self):
        trop_grand = b"x" * (settings.MAX_IMAGE_SIZE_MB * 1024 * 1024 + 1)
        with pytest.raises(HTTPException) as exc:
            prediction_service.process_image(trop_grand, filename="big.png")
        assert exc.value.status_code == 413

    def test_process_audio_vide_422(self):
        with pytest.raises(HTTPException) as exc:
            prediction_service.process_audio(b"", filename="test.mp3")
        assert exc.value.status_code == 422

    def test_process_audio_trop_grand_413(self):
        trop_grand = b"x" * (settings.MAX_AUDIO_SIZE_MB * 1024 * 1024 + 1)
        with pytest.raises(HTTPException) as exc:
            prediction_service.process_audio(trop_grand, filename="big.mp3")
        assert exc.value.status_code == 413

    def test_process_auto_type_non_supporte_415(self):
        with pytest.raises(HTTPException) as exc:
            prediction_service.process_auto(
                file_bytes=b"data",
                filename="archive.tar.gz",
                content_type="application/gzip",
            )
        assert exc.value.status_code == 415

    def test_process_auto_texte_fichier(self):
        txt_bytes = "Avis client très satisfaisant dans un fichier texte.".encode("utf-8")
        resp = prediction_service.process_auto(
            file_bytes=txt_bytes,
            filename="avis.txt",
            content_type="text/plain",
        )
        assert resp.source == "text"
        assert resp.sentiment_predit in {"positif", "negatif", "neutre"}


# ═══════════════════════════════════════════════════════════════════════════════
#  5. TESTS DU MODULE OCR AVEC MOCK (modules/ocr.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOCRModule:
    """Tests unitaires du prétraitement et de l'extraction OCR."""

    def test_nettoyer_texte_ocr(self):
        brut = "AVIS   CLIENT\n\nLivraison | express @ boutique"
        net = ocr_mod._nettoyer_texte_ocr(brut)
        assert "\n" not in net
        assert "AVIS CLIENT" in net

    def test_nettoyer_texte_ocr_vide(self):
        assert ocr_mod._nettoyer_texte_ocr("") == ""

    @patch("pytesseract.image_to_string")
    def test_extraire_texte_image_avec_mock(self, mock_tess):
        from PIL import Image
        mock_tess.return_value = "AVIS CLIENT VÉRIFIÉ Produit parfait"

        # Création d'une image mémoire de test
        img = Image.new("RGB", (300, 100), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        texte = ocr_mod.extraire_texte_image(raw_bytes)
        assert "Produit parfait" in texte
        mock_tess.assert_called_once()

    @patch("pytesseract.image_to_string")
    def test_extraire_texte_image_sans_texte(self, mock_tess):
        from PIL import Image
        mock_tess.return_value = ""  # Aucun texte détecté

        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        texte = ocr_mod.extraire_texte_image(buf.getvalue())
        assert texte == ""


# ═══════════════════════════════════════════════════════════════════════════════
#  6. TESTS DU MODULE ASR AVEC MOCK (modules/asr.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestASRModule:
    """Tests unitaires du module de transcription Whisper."""

    def test_nettoyer_texte_asr(self):
        brut = "[00:00.000 --> 00:04.000] Euh... bonjour, le produit est parfait"
        net = asr_mod._nettoyer_texte_asr(brut)
        assert "-->" not in net
        assert "bonjour" in net

    def test_nettoyer_texte_asr_vide(self):
        assert asr_mod._nettoyer_texte_asr("") == ""

    def test_transcrire_audio_avec_mock(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Bonjour, très satisfait de ma commande."}

        with patch.object(asr_mod, "_get_modele_whisper", return_value=mock_model):
            texte = asr_mod.transcrire_audio(b"fake_mp3_bytes_here")
            assert "très satisfait" in texte

    def test_transcrire_audio_silencieux_retourne_vide(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": ""}

        with patch.object(asr_mod, "_get_modele_whisper", return_value=mock_model):
            texte = asr_mod.transcrire_audio(b"fake_silent_bytes")
            assert texte == ""


# ═══════════════════════════════════════════════════════════════════════════════
#  7. TESTS DE CONFIGURATION & HEALTH (core/config.py & main.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigAndHealth:
    """Tests d'infrastructure et configuration."""

    def test_health_endpoint(self):
        from main import health_check
        h = health_check()
        assert h.status == "ok"
        assert h.model_loaded is True
        assert h.version == "1.0.0"

    def test_model_files_exist(self):
        assert settings.MODEL_PATH.exists()
        assert settings.VECTORIZER_PATH.exists()

    def test_allowed_mimes(self):
        assert "image/png" in settings.ALLOWED_IMAGE_MIMES
        assert "audio/mpeg" in settings.ALLOWED_AUDIO_MIMES
