"""routers module."""
from routers.text import router as text_router
from routers.image import router as image_router
from routers.audio import router as audio_router
from routers.auto import router as auto_router

__all__ = ["text_router", "image_router", "audio_router", "auto_router"]
