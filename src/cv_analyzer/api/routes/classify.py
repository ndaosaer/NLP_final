"""Routes de classification de CV."""

from fastapi import APIRouter, HTTPException

from ..schemas import (
    TextInput,
    ClassificationResult,
    ClassificationProbaResult,
    CategoriesResponse,
)
from ..dependencies import ClassifierService

router = APIRouter(prefix="/api/v1", tags=["Classification"])


@router.post("/classify", response_model=ClassificationResult)
async def classify_cv(input_data: TextInput):
    """
    Classifie un CV et retourne la catégorie prédite.

    Args:
        input_data: Texte du CV à classifier

    Returns:
        Catégorie prédite avec confiance optionnelle
    """
    classifier = ClassifierService.get_instance()

    if not classifier:
        raise HTTPException(
            status_code=503,
            detail="Classificateur non disponible. Le modèle n'a pas été chargé."
        )

    try:
        category = classifier.predict(input_data.text)

        # Essayer d'obtenir la confiance
        confidence = None
        try:
            probas = classifier.predict_proba(input_data.text)
            confidence = probas.get(category)
        except (ValueError, AttributeError):
            pass

        return ClassificationResult(
            category=category,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de classification: {str(e)}")


@router.post("/classify/proba", response_model=ClassificationProbaResult)
async def classify_cv_proba(input_data: TextInput):
    """
    Classifie un CV et retourne les probabilités pour chaque catégorie.

    Args:
        input_data: Texte du CV à classifier

    Returns:
        Catégorie prédite avec probabilités
    """
    classifier = ClassifierService.get_instance()

    if not classifier:
        raise HTTPException(
            status_code=503,
            detail="Classificateur non disponible. Le modèle n'a pas été chargé."
        )

    try:
        category = classifier.predict(input_data.text)
        probabilities = classifier.predict_proba(input_data.text)

        return ClassificationProbaResult(
            category=category,
            probabilities=probabilities
        )
    except ValueError as e:
        if "predict_proba" in str(e):
            raise HTTPException(
                status_code=400,
                detail="Le modèle actuel ne supporte pas les probabilités"
            )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de classification: {str(e)}")


@router.get("/categories", response_model=CategoriesResponse)
async def get_categories():
    """
    Retourne la liste des catégories disponibles.

    Returns:
        Liste des catégories et leur nombre
    """
    categories = ClassifierService.get_categories()

    if not categories:
        raise HTTPException(
            status_code=503,
            detail="Classificateur non disponible. Aucune catégorie disponible."
        )

    return CategoriesResponse(
        categories=categories,
        count=len(categories)
    )
