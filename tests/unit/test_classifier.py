"""Tests unitaires pour le module classifier."""

import pytest
from unittest.mock import Mock, patch
from cv_analyzer.classifier import CVClassifier


class TestCVClassifier:
    """Tests pour CVClassifier."""

    @pytest.fixture
    def classifier(self):
        """Classificateur non entraine."""
        return CVClassifier(model_type='naive_bayes', max_features=100)

    def test_initialization(self, classifier):
        """Test initialisation."""
        assert classifier.model_type == 'naive_bayes'
        assert classifier.max_features == 100
        assert not classifier._is_trained

    def test_available_models(self):
        """Test modeles disponibles."""
        assert 'naive_bayes' in CVClassifier.AVAILABLE_MODELS
        assert 'logistic_regression' in CVClassifier.AVAILABLE_MODELS
        assert 'svm' in CVClassifier.AVAILABLE_MODELS
        assert 'random_forest' in CVClassifier.AVAILABLE_MODELS

    def test_invalid_model_type(self):
        """Test type de modele invalide."""
        with pytest.raises(ValueError):
            CVClassifier(model_type='invalid_model')

    def test_predict_without_training(self, classifier):
        """Test prediction sans entrainement."""
        with pytest.raises(ValueError, match="entraine"):
            classifier.predict("Some CV text here")

    def test_predict_proba_without_training(self, classifier):
        """Test predict_proba sans entrainement."""
        with pytest.raises(ValueError, match="entraine"):
            classifier.predict_proba("Some CV text here")

    def test_preprocess_documents(self, classifier):
        """Test preprocessing des documents."""
        texts = ["Hello world test", "Another document here"]
        docs = classifier.preprocess_documents(texts)

        assert len(docs) == 2
        assert all(isinstance(d, list) for d in docs)

    def test_encode_decode_labels(self, classifier):
        """Test encodage/decodage des labels."""
        import numpy as np

        labels = np.array(['Cat1', 'Cat2', 'Cat1', 'Cat3'])
        encoded = classifier._encode_labels(labels)

        assert len(encoded) == 4
        assert len(classifier.label_to_idx) == 3

        decoded = classifier._decode_labels(encoded)
        assert list(decoded) == list(labels)

    def test_different_model_types(self):
        """Test creation de differents modeles."""
        for model_type in CVClassifier.AVAILABLE_MODELS:
            classifier = CVClassifier(model_type=model_type)
            assert classifier.model_type == model_type
            assert classifier.classifier is not None


class TestCVClassifierLoaded:
    """Tests avec classificateur charge."""

    @pytest.fixture
    def loaded_classifier(self, model_path, classifier_loaded):
        """Classificateur charge depuis fichier."""
        if classifier_loaded:
            return CVClassifier.load(model_path)
        pytest.skip("Modele non disponible")

    def test_predict(self, loaded_classifier, sample_cv_data_scientist):
        """Test prediction."""
        category = loaded_classifier.predict(sample_cv_data_scientist)

        assert isinstance(category, str)
        assert len(category) > 0

    def test_predict_proba(self, loaded_classifier, sample_cv_data_scientist):
        """Test prediction avec probabilites."""
        try:
            probas = loaded_classifier.predict_proba(sample_cv_data_scientist)
            assert isinstance(probas, dict)
            assert len(probas) > 0
            assert all(0 <= p <= 1 for p in probas.values())
        except ValueError:
            # Certains modeles ne supportent pas predict_proba
            pytest.skip("Modele ne supporte pas predict_proba")

    def test_get_categories(self, loaded_classifier):
        """Test obtention des categories."""
        categories = list(loaded_classifier.label_to_idx.keys())

        assert len(categories) > 0
        assert all(isinstance(c, str) for c in categories)

    def test_is_trained(self, loaded_classifier):
        """Test flag is_trained."""
        assert loaded_classifier._is_trained is True
