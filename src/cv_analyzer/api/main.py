"""Point d entree de l API."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import urllib.request
from pathlib import Path
from .dependencies import get_settings, ClassifierService
from .routes import health, classify, summarize, analyze

def train_model(model_path):
    import pandas as pd
    from ..classifier import CVClassifier
    url = "https://raw.githubusercontent.com/florex/resume_corpus/master/resumes.csv"
    dst = Path(model_path).parent.parent / "raw" / "cv_dataset.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dst)
        df = pd.read_csv(dst)
        clf = CVClassifier(model_type="naive_bayes", max_features=1500)
        clf.train(str(dst), text_column=df.columns[0], category_column=df.columns[1])
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        clf.save(model_path)
        return True
    except Exception as e:
        print(f"Erreur: {e}")
        return False

@asynccontextmanager
async def lifespan(app):
    settings = get_settings()
    print(f"Demarrage {settings.app_name} v{settings.version}")
    if not Path(settings.model_path).exists():
        print("Entrainement du modele...")
        train_model(settings.model_path)
    if ClassifierService.load(settings.model_path):
        print(f"Classificateur charge avec {len(ClassifierService.get_categories())} categories")
    else:
        print("Classificateur non charge")
    yield
    print("Arret")

def create_app():
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan, docs_url="/docs", redoc_url="/redoc")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.include_router(health.router)
    app.include_router(classify.router)
    app.include_router(summarize.router)
    app.include_router(analyze.router)
    frontend = os.path.join(os.path.dirname(__file__), "../../../../frontend")
    if os.path.exists(frontend):
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app

app = create_app()
