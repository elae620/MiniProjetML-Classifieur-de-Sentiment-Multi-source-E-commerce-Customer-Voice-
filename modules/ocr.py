"""
modules/ocr.py — Extraction de texte depuis une image (OCR).

Pipeline image → texte :
  1. Ouverture via Pillow
  2. Prétraitement : niveaux de gris, redimensionnement, binarisation adaptative
  3. Extraction du texte via pytesseract (langue : fra)
  4. Nettoyage minimal du texte extrait
  5. Fallback gracieux si l'image ne contient pas de texte lisible

Utilisation dans la cascade :
    texte = extraire_texte_image(chemin_ou_bytes)
    label, conf = classifier.predict(texte)
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Union

log = logging.getLogger(__name__)

# Types acceptés en entrée
ImageInput = Union[str, Path, bytes, io.IOBase]

# ── Constantes de prétraitement ───────────────────────────────────────────────
_LARGEUR_CIBLE = 1200          # px — Tesseract performe mieux sur images larges
_SEUIL_TEXTE_MIN = 10          # Nb de caractères minimum pour considérer un texte valide
_LANG_TESSERACT = "fra"        # Langue française pour Tesseract
_CONFIG_TESSERACT = "--psm 3"  # Page Segmentation Mode 3 : détection automatique


def _charger_image(source: ImageInput):
    """
    Charge une image depuis un chemin, bytes ou flux IO.

    Args:
        source: Chemin (str/Path), bytes bruts ou flux IO.

    Returns:
        Image PIL.Image.Image en mode RGB.

    Raises:
        ImportError   : Si Pillow n'est pas installé.
        ValueError    : Si la source est invalide.
        FileNotFoundError: Si le fichier est introuvable.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow requis : pip install Pillow") from e

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image introuvable : {path}")
        img = Image.open(path)
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, io.IOBase):
        img = Image.open(source)
    else:
        raise ValueError(f"Type source non supporté : {type(source)}")

    return img.convert("RGB")


def _pretraiter_image(img):
    """
    Applique le pipeline de prétraitement pour améliorer l'OCR :
    1. Conversion en niveaux de gris
    2. Redimensionnement si trop petite (Tesseract préfère >= 1200px de large)
    3. Binarisation adaptative (seuillage d'Otsu via PIL)

    Args:
        img: PIL.Image.Image en RGB.

    Returns:
        PIL.Image.Image en niveaux de gris, prête pour Tesseract.
    """
    from PIL import Image, ImageFilter, ImageOps

    # 1. Niveaux de gris
    img_gray = img.convert("L")

    # 2. Redimensionnement si nécessaire
    w, h = img_gray.size
    if w < _LARGEUR_CIBLE:
        ratio    = _LARGEUR_CIBLE / w
        new_size = (int(w * ratio), int(h * ratio))
        img_gray = img_gray.resize(new_size, Image.LANCZOS)
        log.debug("Image redimensionnée : %s → %s", (w, h), new_size)

    # 3. Amélioration du contraste + binarisation (seuil global Otsu simplifié)
    img_gray = ImageOps.autocontrast(img_gray, cutoff=2)

    # 4. Légère débruitage (filtre médian)
    img_gray = img_gray.filter(ImageFilter.MedianFilter(size=3))

    return img_gray


def extraire_texte_image(source: ImageInput) -> str:
    """
    Extrait le texte d'une image via OCR (pytesseract + Pillow).

    Args:
        source: Chemin de fichier (str/Path), bytes bruts, ou flux IO.

    Returns:
        Texte extrait et nettoyé. Chaîne vide si aucun texte détecté.

    Raises:
        ImportError      : Si pytesseract ou Pillow ne sont pas installés.
        FileNotFoundError: Si l'image est introuvable.
        RuntimeError     : Si Tesseract n'est pas accessible sur le système.
    """
    # Import tardif pour permettre le mock dans les tests
    try:
        import pytesseract  # type: ignore
        # Auto-détection de Tesseract sous Windows si installé dans Program Files
        tess_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if tess_default.exists():
            pytesseract.pytesseract.tesseract_cmd = str(tess_default)
    except ImportError as e:
        raise ImportError(
            "pytesseract requis : pip install pytesseract\n"
            "Et Tesseract-OCR doit être installé sur le système."
        ) from e

    # Chargement et prétraitement
    img     = _charger_image(source)
    img_pre = _pretraiter_image(img)

    # OCR
    try:
        texte_brut = pytesseract.image_to_string(
            img_pre,
            lang=_LANG_TESSERACT,
            config=_CONFIG_TESSERACT,
        )
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract-OCR non trouvé sur ce système.\n"
            "Windows : https://github.com/UB-Mannheim/tesseract/wiki\n"
            "Linux   : sudo apt install tesseract-ocr tesseract-ocr-fra\n"
            "Mac     : brew install tesseract"
        ) from e

    # Nettoyage du texte OCR
    texte = _nettoyer_texte_ocr(texte_brut)

    if len(texte) < _SEUIL_TEXTE_MIN:
        log.warning("OCR : texte extrait trop court ('%s') — image peut-être sans texte lisible", texte)
        return ""

    log.info("OCR réussi : %d caractères extraits", len(texte))
    return texte


def _nettoyer_texte_ocr(texte: str) -> str:
    """
    Nettoie le texte brut sorti de Tesseract.

    - Supprime les caractères parasites OCR fréquents
    - Normalise les espaces et sauts de ligne
    - Conserve la ponctuation utile

    Args:
        texte: Texte brut de Tesseract.

    Returns:
        Texte nettoyé.
    """
    if not texte:
        return ""

    # Remplacement des sauts de ligne multiples par un espace
    texte = re.sub(r"\n+", " ", texte)

    # Suppression des caractères parasites OCR courants (|, ___, etc.)
    texte = re.sub(r"[|_]{2,}", " ", texte)

    # Suppression des suites de caractères spéciaux non-textuels
    texte = re.sub(r"[^\w\s\'\-\.\,\!\?\:\;\@\#\€\%]", " ", texte)

    # Normalisation des espaces multiples
    texte = re.sub(r"\s+", " ", texte)

    return texte.strip()
