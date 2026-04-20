"""Tests d'integration pour l'API FastAPI."""

import pytest
from fastapi.testclient import TestClient

from cv_analyzer.api.main import app


class TestHealthEndpoints:
    """Tests pour les endpoints de sante."""

    @pytest.fixture
    def client(self):
        """Client de test."""
        return TestClient(app)

    def test_health_check(self, client):
        """Test endpoint /health."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_info_endpoint(self, client):
        """Test endpoint /info."""
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "classifier_loaded" in data
        assert "summarize_methods" in data
        assert isinstance(data["summarize_methods"], list)


class TestClassificationEndpoints:
    """Tests pour les endpoints de classification."""

    @pytest.fixture
    def client(self):
        """Client de test."""
        return TestClient(app)

    def test_classify_valid_text(self, client, sample_cv_data_scientist, classifier_loaded):
        """Test classification avec texte valide."""
        response = client.post(
            "/api/v1/classify",
            json={"text": sample_cv_data_scientist}
        )

        if classifier_loaded:
            assert response.status_code == 200
            data = response.json()
            assert "category" in data
            assert isinstance(data["category"], str)
        else:
            # 503 si classificateur non charge
            assert response.status_code == 503

    def test_classify_short_text(self, client, sample_cv_short):
        """Test classification avec texte trop court."""
        response = client.post(
            "/api/v1/classify",
            json={"text": sample_cv_short}
        )

        # Should fail validation (min_length=50)
        assert response.status_code == 422

    def test_classify_proba(self, client, sample_cv_data_scientist):
        """Test classification avec probabilites."""
        response = client.post(
            "/api/v1/classify/proba",
            json={"text": sample_cv_data_scientist}
        )

        # May return 200 or 400 depending on model type
        if response.status_code == 200:
            data = response.json()
            assert "category" in data
            assert "probabilities" in data
            assert isinstance(data["probabilities"], dict)

    def test_get_categories(self, client):
        """Test liste des categories."""
        response = client.get("/api/v1/categories")

        # May be 200 or 503 if classifier not loaded
        if response.status_code == 200:
            data = response.json()
            assert "categories" in data
            assert "count" in data
            assert isinstance(data["categories"], list)


class TestSummarizationEndpoints:
    """Tests pour les endpoints de resume."""

    @pytest.fixture
    def client(self):
        """Client de test."""
        return TestClient(app)

    def test_summarize_valid_text(self, client, sample_cv_data_scientist):
        """Test resume avec texte valide."""
        response = client.post(
            "/api/v1/summarize",
            json={
                "text": sample_cv_data_scientist,
                "method": "tfidf",
                "num_sentences": 3
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "sentences" in data
        assert "method" in data
        assert "word_count" in data
        assert len(data["sentences"]) <= 3

    def test_summarize_all_methods(self, client, sample_cv_data_scientist):
        """Test toutes les methodes de resume."""
        for method in ["tfidf", "textrank", "frequency"]:
            response = client.post(
                "/api/v1/summarize",
                json={
                    "text": sample_cv_data_scientist,
                    "method": method,
                    "num_sentences": 3
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["method"] == method

    def test_summarize_invalid_method(self, client, sample_cv_data_scientist):
        """Test methode invalide."""
        response = client.post(
            "/api/v1/summarize",
            json={
                "text": sample_cv_data_scientist,
                "method": "invalid_method",
                "num_sentences": 3
            }
        )

        assert response.status_code == 400

    def test_get_summarize_methods(self, client):
        """Test liste des methodes."""
        response = client.get("/api/v1/summarize/methods")

        assert response.status_code == 200
        data = response.json()
        assert "methods" in data
        assert "tfidf" in data["methods"]
        assert "textrank" in data["methods"]
        assert "frequency" in data["methods"]


class TestAnalysisEndpoints:
    """Tests pour les endpoints d'analyse complete."""

    @pytest.fixture
    def client(self):
        """Client de test."""
        return TestClient(app)

    def test_analyze_full(self, client, sample_cv_data_scientist):
        """Test analyse complete."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "text": sample_cv_data_scientist,
                "classify": True,
                "summarize": True,
                "method": "tfidf",
                "num_sentences": 3
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert data["stats"]["words"] > 0
        assert data["stats"]["characters"] > 0

    def test_analyze_summary_only(self, client, sample_cv_data_scientist):
        """Test analyse sans classification."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "text": sample_cv_data_scientist,
                "classify": False,
                "summarize": True,
                "num_sentences": 3
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["classification"] is None
        assert data["summary"] is not None

    def test_analyze_classify_only(self, client, sample_cv_data_scientist):
        """Test analyse sans resume."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "text": sample_cv_data_scientist,
                "classify": True,
                "summarize": False,
                "num_sentences": 3
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["summary"] is None

    def test_analyze_stats(self, client, sample_cv_data_scientist):
        """Test statistiques."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "text": sample_cv_data_scientist,
                "classify": False,
                "summarize": False
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "words" in data["stats"]
        assert "characters" in data["stats"]
        assert "sentences" in data["stats"]
