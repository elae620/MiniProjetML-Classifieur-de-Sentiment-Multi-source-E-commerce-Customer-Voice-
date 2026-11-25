"""
main.py — Point d'entrée principal de l'API REST FastAPI.

Architecture en cascade pour l'analyse de sentiment multi-source :
  - POST /predict/text   : Avis texte direct
  - POST /predict/image  : Image (OCR -> Modèle texte)
  - POST /predict/audio  : Audio (ASR Whisper -> Modèle texte)
  - POST /predict        : Détection automatique du média
  - GET  /health         : Statut de santé et état des modèles
  - GET  /docs & /redoc  : Documentation Swagger / OpenAPI
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from modules.classifier import _charger_modele
from routers import text_router, image_router, audio_router, auto_router
from schemas.prediction import HealthResponse

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")


# ── Gestion du cycle de vie (Lifespan) ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pré-chargement des modèles au démarrage du serveur.
    Garantit que le modèle est prêt en mémoire avant de traiter la première requête.
    """
    log.info("Démarrage de l'API : pré-chargement des modèles...")
    try:
        # Chargement du modèle de classification textuel
        modele = _charger_modele(str(settings.MODEL_PATH))
        log.info("✓ Modèle textuel chargé avec succès (%d classes).", len(modele.classes_))
    except Exception as e:
        log.error("✗ Impossible de charger le modèle au démarrage : %s", e)
        log.warning("Veillez à exécuter 'python train.py' pour générer les poids.")

    yield
    log.info("Arrêt de l'API : libération des ressources.")


# ── Initialisation FastAPI ───────────────────────────────────────────────────
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description=(
        "API REST intelligente pour la classification de sentiment multi-source "
        "(Texte, Image OCR, Audio ASR) selon une architecture en cascade.\n\n"
        "**Livrable d'examen — Mini-Projet ML Appliqué 2026**."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware CORS ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Gestionnaires d'erreurs personnalisés ────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Capture les erreurs de validation Pydantic et renvoie un statut 422 standardisé.
    """
    erreurs = []
    for err in exc.errors():
        champ = " -> ".join(str(loc) for loc in err.get("loc", []))
        message = err.get("msg", "Erreur de validation")
        erreurs.append(f"{champ}: {message}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Échec de validation de la charge utile (422 Unprocessable Entity)",
            "erreurs": erreurs,
        },
    )


# ── Route de santé ────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Système"],
    summary="Vérification de l'état de l'API",
)
def health_check() -> HealthResponse:
    """
    Renvoie le statut opérationnel de l'API et la disponibilité du modèle.
    """
    model_ok = settings.MODEL_PATH.exists()
    return HealthResponse(
        status="ok" if model_ok else "degraded",
        model_loaded=model_ok,
        version=settings.PROJECT_VERSION,
    )


# ── Inclusion des routeurs ────────────────────────────────────────────────────
app.include_router(text_router)
app.include_router(image_router)
app.include_router(audio_router)
app.include_router(auto_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
