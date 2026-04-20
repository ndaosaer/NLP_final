"""Schemas Pydantic pour l'API CV Analyzer."""

from typing import Optional
from pydantic import BaseModel, Field


# --- Input Schemas ---

class TextInput(BaseModel):
    """Input pour analyse de texte."""
    text: str = Field(..., min_length=50, description="Texte du CV (minimum 50 caractères)")


class SummarizeInput(BaseModel):
    """Input pour résumé de CV."""
    text: str = Field(..., min_length=50, description="Texte du CV")
    method: str = Field(default="tfidf", description="Méthode: tfidf, textrank, frequency")
    num_sentences: int = Field(default=5, ge=1, le=10, description="Nombre de phrases")


class AnalyzeInput(BaseModel):
    """Input pour analyse complète."""
    text: str = Field(..., min_length=50, description="Texte du CV")
    classify: bool = Field(default=True, description="Effectuer la classification")
    summarize: bool = Field(default=True, description="Générer un résumé")
    method: str = Field(default="tfidf", description="Méthode de résumé")
    num_sentences: int = Field(default=5, ge=1, le=10, description="Nombre de phrases")


# --- Output Schemas ---

class HealthResponse(BaseModel):
    """Réponse health check."""
    status: str
    version: str


class InfoResponse(BaseModel):
    """Informations sur l'API."""
    name: str
    version: str
    classifier_loaded: bool
    classifier_categories: list[str]
    summarize_methods: list[str]


class ExtractTextResponse(BaseModel):
    """Réponse extraction de texte."""
    text: str
    length: int
    format: str


class ClassificationResult(BaseModel):
    """Résultat de classification."""
    category: str
    confidence: Optional[float] = None


class ClassificationProbaResult(BaseModel):
    """Résultat de classification avec probabilités."""
    category: str
    probabilities: dict[str, float]


class CategoriesResponse(BaseModel):
    """Liste des catégories disponibles."""
    categories: list[str]
    count: int


class SummaryResult(BaseModel):
    """Résultat de résumé."""
    summary: str
    sentences: list[str]
    method: str
    word_count: int


class SummarizeMethodsResponse(BaseModel):
    """Méthodes de résumé disponibles."""
    methods: list[str]


class TextStats(BaseModel):
    """Statistiques sur le texte."""
    words: int
    characters: int
    sentences: int


class AnalysisResult(BaseModel):
    """Résultat d'analyse complète."""
    classification: Optional[ClassificationResult] = None
    summary: Optional[SummaryResult] = None
    stats: TextStats


# --- Error Schemas ---

class ErrorResponse(BaseModel):
    """Réponse d'erreur."""
    detail: str
    error_code: Optional[str] = None
