"""Point d'entrée de l'API FastAPI CV Analyzer."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import get_settings, ClassifierService
from .routes import health, classify, summarize, analyze


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    # Startup
    settings = get_settings()
    print(f"Démarrage de {settings.app_name} v{settings.version}")

    # Charger le modèle de classification
    if ClassifierService.load(settings.model_path):
        categories = ClassifierService.get_categories()
        print(f"Classificateur chargé avec {len(categories)} catégories")
    else:
        print("Attention: Classificateur non chargé (fonctionnalité réduite)")

    yield

    # Shutdown
    print("Arrêt de l'application")


def create_app() -> FastAPI:
    """Factory pour créer l'application FastAPI."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="API pour l'analyse et la classification de CV",
        version=settings.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En production, spécifier les origines autorisées
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Enregistrer les routes
    app.include_router(health.router)
    app.include_router(classify.router)
    app.include_router(summarize.router)
    app.include_router(analyze.router)

    return app


# Instance de l'application
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cv_analyzer.api.main:app", host="0.0.0.0", port=8000, reload=True)
