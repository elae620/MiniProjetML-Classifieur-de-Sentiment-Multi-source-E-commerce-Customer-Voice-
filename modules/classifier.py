"""
modules/classifier.py — Chargement unique du modèle + fonction de prédiction.

Rôle dans l'architecture en cascade :
  texte brut (depuis OCR ou ASR ou formulaire)
      └─► predict(text) ─► (label: str, confiance: float)

Le modèle est chargé UNE SEULE FOIS au premier appel (singleton via module-level).
Toutes les routes FastAPI partagent la même instance en mémoire.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import joblib
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)

# Chemin par défaut vers le pipeline sérialisé
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"


@lru_cache(maxsize=1)
def _charger_modele(chemin: str = str(_MODEL_PATH)) -> Pipeline:
    """
    Charge le pipeline sklearn depuis le disque (cache LRU → singleton).

    Args:
        chemin: Chemin absolu vers model.pkl.

    Returns:
        Pipeline sklearn entraîné.

    Raises:
        FileNotFoundError: Si le fichier modèle est absent.
        RuntimeError    : Si le chargement échoue.
    """
    path = Path(chemin)
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {path}\n"
            "Lancez d'abord : python train.py"
        )
    try:
        model = joblib.load(path)
        log.info("Modèle chargé depuis %s | classes=%s", path, list(model.classes_))
        return model
    except Exception as exc:
        raise RuntimeError(f"Erreur lors du chargement du modèle : {exc}") from exc


def predict(texte: str, model_path: str | None = None) -> Tuple[str, float]:
    """
    Prédit le sentiment d'un texte brut.

    Args:
        texte      : Texte à classifier (brut, non nettoyé — le pipeline s'en charge).
        model_path : Chemin optionnel vers model.pkl (utile pour les tests).

    Returns:
        Tuple (label, confiance) :
            - label     : "positif" | "negatif" | "neutre"
            - confiance : probabilité max dans [0, 1]

    Raises:
        ValueError: Si le texte est vide après strip.
    """
    texte = (texte or "").strip()
    if not texte:
        raise ValueError("Le texte fourni est vide — impossible de classifier.")

    chemin = model_path or str(_MODEL_PATH)
    model  = _charger_modele(chemin)

    # predict_proba retourne shape (1, n_classes)
    proba  = model.predict_proba([texte])[0]
    idx    = int(proba.argmax())
    label  = model.classes_[idx]
    confiance = round(float(proba[idx]), 4)

    log.debug("predict('%s…') → %s (conf=%.3f)", texte[:40], label, confiance)
    return label, confiance
