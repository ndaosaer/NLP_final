"""Routes d'analyse complète de CV."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from ..schemas import (
    AnalyzeInput,
    AnalysisResult,
    ClassificationResult,
    SummaryResult,
    TextStats,
    ExtractTextResponse,
)
from ..dependencies import ClassifierService, SummarizerService, FileLoaderService

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


def count_sentences(text: str) -> int:
    """Compte approximativement le nombre de phrases."""
    import re
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


@router.post("/extract-text", response_model=ExtractTextResponse)
async def extract_text(file: UploadFile = File(...)):
    """
    Extrait le texte d'un fichier CV (PDF, DOCX, TXT).

    Args:
        file: Fichier uploadé

    Returns:
        Texte extrait avec métadonnées
    """
    # Vérifier l'extension
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()

    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté: {suffix}. Formats acceptés: .pdf, .docx, .txt"
        )

    try:
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extraire le texte
        file_loader = FileLoaderService.get_instance()
        text = file_loader.load(tmp_path)

        # Nettoyer le fichier temporaire
        Path(tmp_path).unlink()

        if not text or len(text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Le fichier ne contient pas assez de texte (minimum 50 caractères)"
            )

        return ExtractTextResponse(
            text=text,
            length=len(text),
            format=suffix[1:]  # Remove the dot
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'extraction: {str(e)}")


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_cv(input_data: AnalyzeInput):
    """
    Analyse complète d'un CV : classification + résumé.

    Args:
        input_data: Texte du CV et options d'analyse

    Returns:
        Résultat complet avec classification, résumé et statistiques
    """
    result = AnalysisResult(
        classification=None,
        summary=None,
        stats=TextStats(
            words=len(input_data.text.split()),
            characters=len(input_data.text),
            sentences=count_sentences(input_data.text)
        )
    )

    # Classification
    if input_data.classify:
        classifier = ClassifierService.get_instance()
        if classifier:
            try:
                category = classifier.predict(input_data.text)
                confidence = None
                try:
                    probas = classifier.predict_proba(input_data.text)
                    confidence = probas.get(category)
                except (ValueError, AttributeError):
                    pass

                result.classification = ClassificationResult(
                    category=category,
                    confidence=confidence
                )
            except Exception as e:
                # Log l'erreur mais continue
                print(f"Erreur classification: {e}")

    # Résumé
    if input_data.summarize:
        try:
            summarizer = SummarizerService.get_instance(input_data.method)
            sentences = summarizer.summarize_as_list(
                input_data.text,
                num_sentences=input_data.num_sentences
            )
            summary_text = summarizer.summarize(
                input_data.text,
                num_sentences=input_data.num_sentences
            )

            result.summary = SummaryResult(
                summary=summary_text,
                sentences=sentences,
                method=input_data.method,
                word_count=len(input_data.text.split())
            )
        except Exception as e:
            print(f"Erreur résumé: {e}")

    return result
