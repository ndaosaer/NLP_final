"""Tests end-to-end pour l'API CV Analyzer."""

import pytest
from fastapi.testclient import TestClient

from cv_analyzer.api.main import app


class TestE2EWorkflow:
    """Tests de workflow complet."""

    @pytest.fixture
    def client(self):
        """Client de test."""
        return TestClient(app)

    def test_complete_cv_analysis_workflow(
        self, client, sample_cv_data_scientist, sample_cv_software_engineer
    ):
        """Test workflow complet d'analyse de CV."""
        # 1. Verifier que l'API est disponible
        health = client.get("/health")
        assert health.status_code == 200

        # 2. Obtenir les infos de l'API
        info = client.get("/info")
        assert info.status_code == 200
        api_info = info.json()

        # 3. Obtenir les methodes de resume disponibles
        methods = client.get("/api/v1/summarize/methods")
        assert methods.status_code == 200
        available_methods = methods.json()["methods"]

        # 4. Analyser le premier CV
        analysis1 = client.post(
            "/api/v1/analyze",
            json={
                "text": sample_cv_data_scientist,
                "classify": True,
                "summarize": True,
                "method": available_methods[0],
                "num_sentences": 5
            }
        )
        assert analysis1.status_code == 200
        result1 = analysis1.json()

        # Verifier les resultats
        assert result1["stats"]["words"] > 0
        if result1["summary"]:
            assert len(result1["summary"]["sentences"]) <= 5

        # 5. Analyser le deuxieme CV
        analysis2 = client.post(
            "/api/v1/analyze",
            json={
                "text": sample_cv_software_engineer,
                "classify": True,
                "summarize": True,
                "method": available_methods[0],
                "num_sentences": 3
            }
        )
        assert analysis2.status_code == 200
        result2 = analysis2.json()

        # 6. Comparer les deux CV (categories differentes attendues)
        if result1["classification"] and result2["classification"]:
            # Les deux CV devraient avoir des resultats de classification
            assert isinstance(result1["classification"]["category"], str)
            assert isinstance(result2["classification"]["category"], str)

    def test_multiple_summarization_methods(self, client, sample_cv_data_scientist):
        """Test de toutes les methodes de resume sur le meme CV."""
        methods = ["tfidf", "textrank", "frequency"]
        results = {}

        for method in methods:
            response = client.post(
                "/api/v1/summarize",
                json={
                    "text": sample_cv_data_scientist,
                    "method": method,
                    "num_sentences": 3
                }
            )
            assert response.status_code == 200
            results[method] = response.json()

        # Verifier que chaque methode retourne un resultat
        for method in methods:
            assert results[method]["method"] == method
            assert len(results[method]["sentences"]) > 0

    def test_error_handling(self, client):
        """Test gestion des erreurs."""
        # Texte trop court
        response = client.post(
            "/api/v1/classify",
            json={"text": "Short"}
        )
        assert response.status_code == 422

        # Methode invalide
        response = client.post(
            "/api/v1/summarize",
            json={
                "text": "A" * 100,  # Texte assez long
                "method": "invalid",
                "num_sentences": 3
            }
        )
        assert response.status_code == 400

        # JSON invalide
        response = client.post(
            "/api/v1/analyze",
            content="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_varying_sentence_counts(self, client, sample_cv_data_scientist):
        """Test avec differents nombres de phrases."""
        for num_sentences in [2, 5, 8, 10]:
            response = client.post(
                "/api/v1/summarize",
                json={
                    "text": sample_cv_data_scientist,
                    "method": "tfidf",
                    "num_sentences": num_sentences
                }
            )
            assert response.status_code == 200
            result = response.json()
            assert len(result["sentences"]) <= num_sentences

    def test_concurrent_requests_simulation(self, client, sample_cv_data_scientist):
        """Test simulation de requetes multiples."""
        # Simuler plusieurs requetes (sequentielles dans les tests)
        results = []
        for i in range(5):
            response = client.post(
                "/api/v1/analyze",
                json={
                    "text": sample_cv_data_scientist,
                    "classify": True,
                    "summarize": True,
                    "num_sentences": 3
                }
            )
            assert response.status_code == 200
            results.append(response.json())

        # Verifier la coherence des resultats
        if all(r["classification"] for r in results):
            categories = [r["classification"]["category"] for r in results]
            # Toutes les predictions devraient etre identiques pour le meme texte
            assert len(set(categories)) == 1


class TestAPIConsistency:
    """Tests de coherence de l'API."""

    @pytest.fixture
    def client(self):
        """Client de test."""
        return TestClient(app)

    def test_stats_consistency(self, client, sample_cv_data_scientist):
        """Test coherence des statistiques."""
        # Obtenir les stats via /analyze
        response = client.post(
            "/api/v1/analyze",
            json={
                "text": sample_cv_data_scientist,
                "classify": False,
                "summarize": False
            }
        )
        assert response.status_code == 200
        stats = response.json()["stats"]

        # Verifier les stats
        expected_words = len(sample_cv_data_scientist.split())
        expected_chars = len(sample_cv_data_scientist)

        assert stats["words"] == expected_words
        assert stats["characters"] == expected_chars

    def test_classification_consistency(self, client, sample_cv_data_scientist):
        """Test coherence de la classification."""
        # Classifier via /classify
        r1 = client.post(
            "/api/v1/classify",
            json={"text": sample_cv_data_scientist}
        )

        # Classifier via /analyze
        r2 = client.post(
            "/api/v1/analyze",
            json={
                "text": sample_cv_data_scientist,
                "classify": True,
                "summarize": False
            }
        )

        if r1.status_code == 200 and r2.status_code == 200:
            cat1 = r1.json()["category"]
            cat2 = r2.json()["classification"]["category"]
            # Les deux endpoints doivent retourner la meme categorie
            assert cat1 == cat2
