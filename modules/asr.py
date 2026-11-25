"""
modules/asr.py — Transcription audio vers texte (ASR).

Pipeline audio → texte :
  1. Validation et normalisation du chemin/bytes
  2. Chargement du modèle Whisper UNE SEULE FOIS (singleton)
  3. Transcription avec langue forcée (fr) pour réduire les erreurs
  4. Nettoyage minimal du texte transcrit
  5. Gestion gracieuse des fichiers corrompus ou silencieux

Utilisation dans la cascade :
    texte = transcrire_audio(chemin_ou_bytes)
    label, conf = classifier.predict(texte)

Note : Whisper accepte nativement MP3, WAV, M4A, FLAC, OGG (via ffmpeg).
       Les fichiers du dataset sont en .mp3 malgré l'extension .wav dans le CSV.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Union

log = logging.getLogger(__name__)

# Types acceptés en entrée
AudioInput = Union[str, Path, bytes, io.IOBase]

# ── Auto-détection de ffmpeg dans le dossier bin local ou imageio_ffmpeg ───────
_LOCAL_BIN = Path(__file__).resolve().parent.parent / "bin"
if _LOCAL_BIN.exists() and str(_LOCAL_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_LOCAL_BIN) + os.pathsep + os.environ.get("PATH", "")
    log.info("Dossier bin local ajouté au PATH pour ffmpeg : %s", _LOCAL_BIN)

# ── Constantes ────────────────────────────────────────────────────────────────
_MODELE_WHISPER   = "base"      # Modèle léger, CPU-friendly (~74 MB, ~1 GB RAM)
_LANGUE_FORCEE    = "fr"        # Forcer le français — évite la détection automatique
_SEUIL_TEXTE_MIN  = 5           # Nb de caractères minimum pour un transcript valide
_EXTENSIONS_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

# ── Singleton du modèle Whisper ───────────────────────────────────────────────
_modele_whisper = None          # Initialisé au premier appel (lazy loading)


def _get_modele_whisper():
    """
    Charge le modèle Whisper en mémoire (singleton lazy).
    Le modèle est téléchargé (~74 MB) au premier appel, puis réutilisé.

    Returns:
        Modèle Whisper chargé.

    Raises:
        ImportError : Si openai-whisper n'est pas installé.
        RuntimeError: Si le chargement échoue.
    """
    global _modele_whisper
    if _modele_whisper is not None:
        return _modele_whisper

    try:
        import whisper  # type: ignore
    except ImportError as e:
        raise ImportError(
            "openai-whisper requis : pip install openai-whisper\n"
            "ffmpeg doit aussi être installé sur le système."
        ) from e

    log.info("Chargement du modèle Whisper '%s'…", _MODELE_WHISPER)
    try:
        _modele_whisper = whisper.load_model(_MODELE_WHISPER)
        log.info("Modèle Whisper '%s' chargé.", _MODELE_WHISPER)
    except Exception as exc:
        raise RuntimeError(f"Échec du chargement Whisper : {exc}") from exc

    return _modele_whisper


def _normaliser_source(source: AudioInput) -> str:
    """
    Normalise la source audio en chemin de fichier temporaire si nécessaire.

    Args:
        source: Chemin (str/Path), bytes bruts ou flux IO.

    Returns:
        Chemin absolu vers un fichier audio (temp si bytes/IO).

    Raises:
        FileNotFoundError: Si le fichier est introuvable.
        ValueError       : Si le type source est invalide ou l'extension non supportée.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        # Gestion de la discordance .wav↔.mp3 : cherche la variante si absent
        if not path.exists():
            variantes = [path.with_suffix(ext) for ext in _EXTENSIONS_AUDIO]
            for v in variantes:
                if v.exists():
                    log.warning(
                        "Fichier '%s' absent, utilisation de la variante '%s'",
                        path.name, v.name,
                    )
                    path = v
                    break
            else:
                raise FileNotFoundError(
                    f"Fichier audio introuvable : {path}\n"
                    f"Variantes testées : {[v.name for v in variantes]}"
                )
        ext = path.suffix.lower()
        if ext not in _EXTENSIONS_AUDIO:
            raise ValueError(
                f"Extension non supportée : '{ext}'. "
                f"Extensions acceptées : {_EXTENSIONS_AUDIO}"
            )
        return str(path.resolve())

    elif isinstance(source, bytes):
        # Écriture dans un fichier temporaire pour Whisper
        suffix = ".mp3"   # Format le plus courant dans le dataset
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(source)
            return tmp.name

    elif isinstance(source, io.IOBase):
        suffix = ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(source.read())
            return tmp.name

    else:
        raise ValueError(f"Type source non supporté : {type(source)}")


def transcrire_audio(source: AudioInput) -> str:
    """
    Transcrit un fichier audio en texte français (ASR via Whisper).

    Args:
        source: Chemin de fichier (str/Path), bytes bruts ou flux IO.
                Formats supportés : WAV, MP3, M4A, FLAC, OGG.
                Si le fichier est en .wav mais stocké en .mp3,
                la variante .mp3 est cherchée automatiquement.

    Returns:
        Texte transcrit et nettoyé.
        Chaîne vide si audio silencieux ou transcription trop courte.

    Raises:
        ImportError      : Si openai-whisper n'est pas installé.
        FileNotFoundError: Si le fichier audio est introuvable.
        RuntimeError     : Si la transcription échoue (fichier corrompu, etc.).
    """
    # 1. Normalisation de la source
    chemin_tmp = None
    try:
        chemin = _normaliser_source(source)

        # Marquer si fichier temporaire (pour suppression après usage)
        if isinstance(source, (bytes, io.IOBase)):
            chemin_tmp = chemin

        # 2. Transcription
        modele = _get_modele_whisper()
        log.info("Transcription de '%s'…", Path(chemin).name)

        try:
            resultat = modele.transcribe(
                chemin,
                language=_LANGUE_FORCEE,    # Forcer le français
                fp16=False,                 # CPU — pas de demi-précision
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Erreur de transcription pour '{Path(chemin).name}' : {exc}\n"
                "Vérifiez que ffmpeg est installé (requis par Whisper pour les MP3)."
            ) from exc

        texte_brut = resultat.get("text", "").strip()

        # 3. Nettoyage
        texte = _nettoyer_texte_asr(texte_brut)

        if len(texte) < _SEUIL_TEXTE_MIN:
            log.warning(
                "ASR : transcription trop courte ('%s') — audio peut-être silencieux",
                texte,
            )
            return ""

        log.info(
            "ASR réussi : %d caractères → '%s…'",
            len(texte), texte[:50],
        )
        return texte

    finally:
        # Suppression du fichier temporaire si créé
        if chemin_tmp and os.path.exists(chemin_tmp):
            os.unlink(chemin_tmp)


def _nettoyer_texte_asr(texte: str) -> str:
    """
    Nettoie la transcription brute de Whisper.

    - Supprime les balises de timing Whisper ([00:00.000 --> 00:05.000])
    - Normalise les espaces
    - Supprime les hésitations courantes (euh, mmh)

    Args:
        texte: Transcription brute de Whisper.

    Returns:
        Texte nettoyé.
    """
    if not texte:
        return ""

    # Suppression des balises de timing (si mode verbose)
    texte = re.sub(r"\[\d+:\d+\.\d+\s*-->\s*\d+:\d+\.\d+\]", "", texte)

    # Suppression des hésitations courantes
    texte = re.sub(r"\b(euh+|mmh+|hm+|ah+)\b", "", texte, flags=re.IGNORECASE)

    # Normalisation des espaces
    texte = re.sub(r"\s+", " ", texte)

    return texte.strip()
