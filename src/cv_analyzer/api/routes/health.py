"""Routes health check et info."""

from fastapi import APIRouter

from ..schemas import HealthResponse, InfoResponse
from ..dependencies import get_settings, ClassifierService, SummarizerService

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérifie que l'API est opérationnelle."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.version
    )


@router.get("/info", response_model=InfoResponse)
async def get_info():
    """Retourne les informations sur l'API."""
    settings = get_settings()
    return InfoResponse(
        name=settings.app_name,
        version=settings.version,
        classifier_loaded=ClassifierService.is_loaded(),
        classifier_categories=ClassifierService.get_categories(),
        summarize_methods=SummarizerService.get_methods()
    )
