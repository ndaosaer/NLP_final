"""Dépendances et injection pour l'API CV Analyzer."""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from ..classifier import CVClassifier
from ..summarizer import CVSummarizer
from ..file_loader import FileLoader


class Settings:
    """Configuration de l'application."""

    def __init__(self):
        self.app_name = "CV Analyzer API"
        self.version = "1.0.0"
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

        # Chemin vers le modèle de classification
        self.model_path = os.getenv(
            "MODEL_PATH",
            str(Path(__file__).parent.parent.parent.parent / "data" / "models" / "classifier.pkl")
        )

        # Configuration du summarizer
        self.default_summarize_method = os.getenv("SUMMARIZE_METHOD", "tfidf")
        self.default_num_sentences = int(os.getenv("NUM_SENTENCES", "5"))


@lru_cache
def get_settings() -> Settings:
    """Retourne les settings (singleton)."""
    return Settings()


class ClassifierService:
    """Service singleton pour le classificateur."""

    _instance: Optional[CVClassifier] = None
    _is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> Optional[CVClassifier]:
        """Retourne l'instance du classificateur."""
        return cls._instance

    @classmethod
    def is_loaded(cls) -> bool:
        """Vérifie si le classificateur est chargé."""
        return cls._is_loaded and cls._instance is not None

    @classmethod
    def load(cls, model_path: str) -> bool:
        """Charge le modèle de classification."""
        try:
            if Path(model_path).exists():
                cls._instance = CVClassifier.load(model_path)
                cls._is_loaded = True
                return True
            else:
                print(f"Modèle non trouvé: {model_path}")
                return False
        except Exception as e:
            print(f"Erreur chargement modèle: {e}")
            cls._is_loaded = False
            return False

    @classmethod
    def get_categories(cls) -> list[str]:
        """Retourne les catégories disponibles."""
        if cls._instance and cls._is_loaded:
            return list(cls._instance.label_to_idx.keys())
        return []


class SummarizerService:
    """Service pour les summarizers."""

    AVAILABLE_METHODS = ["tfidf", "textrank", "frequency"]
    _instances: dict[str, CVSummarizer] = {}

    @classmethod
    def get_instance(cls, method: str = "tfidf") -> CVSummarizer:
        """Retourne une instance de summarizer pour la méthode donnée."""
        if method not in cls.AVAILABLE_METHODS:
            raise ValueError(f"Méthode inconnue: {method}. Disponibles: {cls.AVAILABLE_METHODS}")

        if method not in cls._instances:
            cls._instances[method] = CVSummarizer(method=method)

        return cls._instances[method]

    @classmethod
    def get_methods(cls) -> list[str]:
        """Retourne les méthodes disponibles."""
        return cls.AVAILABLE_METHODS


class FileLoaderService:
    """Service singleton pour le chargement de fichiers."""

    _instance: Optional[FileLoader] = None

    @classmethod
    def get_instance(cls) -> FileLoader:
        """Retourne l'instance du file loader."""
        if cls._instance is None:
            cls._instance = FileLoader()
        return cls._instance


# Fonctions de dépendance pour FastAPI

def get_classifier() -> Optional[CVClassifier]:
    """Dépendance FastAPI pour obtenir le classificateur."""
    return ClassifierService.get_instance()


def get_summarizer(method: str = "tfidf") -> CVSummarizer:
    """Dépendance FastAPI pour obtenir un summarizer."""
    return SummarizerService.get_instance(method)


def get_file_loader() -> FileLoader:
    """Dépendance FastAPI pour obtenir le file loader."""
    return FileLoaderService.get_instance()
