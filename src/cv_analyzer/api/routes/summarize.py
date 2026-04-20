"""Routes de résumé de CV."""

from fastapi import APIRouter, HTTPException

from ..schemas import (
    SummarizeInput,
    SummaryResult,
    SummarizeMethodsResponse,
)
from ..dependencies import SummarizerService

router = APIRouter(prefix="/api/v1", tags=["Summarization"])


@router.post("/summarize", response_model=SummaryResult)
async def summarize_cv(input_data: SummarizeInput):
    """
    Génère un résumé extractif du CV.

    Args:
        input_data: Texte du CV et paramètres de résumé

    Returns:
        Résumé avec phrases extraites et statistiques
    """
    try:
        summarizer = SummarizerService.get_instance(input_data.method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        # Générer le résumé
        sentences = summarizer.summarize_as_list(
            input_data.text,
            num_sentences=input_data.num_sentences
        )

        summary_text = summarizer.summarize(
            input_data.text,
            num_sentences=input_data.num_sentences
        )

        # Compter les mots
        word_count = len(input_data.text.split())

        return SummaryResult(
            summary=summary_text,
            sentences=sentences,
            method=input_data.method,
            word_count=word_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de résumé: {str(e)}")


@router.get("/summarize/methods", response_model=SummarizeMethodsResponse)
async def get_summarize_methods():
    """
    Retourne les méthodes de résumé disponibles.

    Returns:
        Liste des méthodes disponibles
    """
    return SummarizeMethodsResponse(
        methods=SummarizerService.get_methods()
    )
