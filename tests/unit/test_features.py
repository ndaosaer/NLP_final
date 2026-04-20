"""Tests unitaires pour le module features."""

import pytest
import numpy as np
from cv_analyzer.features import TFIDF, BagOfWords, FeatureExtractor, NGramExtractor


class TestTFIDF:
    """Tests pour TFIDF."""

    @pytest.fixture
    def tfidf(self):
        """Instance TFIDF."""
        return TFIDF(max_features=100)

    @pytest.fixture
    def sample_docs(self):
        """Documents d'exemple."""
        return [
            ["python", "machine", "learning", "data"],
            ["java", "software", "engineering", "code"],
            ["python", "data", "science", "analysis"],
        ]

    def test_fit_transform(self, tfidf, sample_docs):
        """Test fit_transform."""
        matrix = tfidf.fit_transform(sample_docs)

        assert matrix is not None
        assert matrix.shape[0] == len(sample_docs)
        assert matrix.shape[1] > 0

    def test_transform_after_fit(self, tfidf, sample_docs):
        """Test transform apres fit."""
        tfidf.fit(sample_docs)
        new_doc = [["python", "data", "new", "term"]]
        matrix = tfidf.transform(new_doc)

        assert matrix is not None
        assert matrix.shape[0] == 1

    def test_get_feature_names(self, tfidf, sample_docs):
        """Test get_feature_names."""
        tfidf.fit(sample_docs)
        names = tfidf.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0
        assert "python" in names or "data" in names

    def test_max_features_limit(self, sample_docs):
        """Test limite max_features."""
        tfidf = TFIDF(max_features=3)
        tfidf.fit(sample_docs)

        assert len(tfidf.get_feature_names()) <= 3

    def test_empty_document(self, tfidf, sample_docs):
        """Test avec document vide."""
        tfidf.fit(sample_docs)
        matrix = tfidf.transform([[]])

        assert matrix is not None
        assert matrix.shape[0] == 1


class TestBagOfWords:
    """Tests pour BagOfWords."""

    @pytest.fixture
    def bow(self):
        """Instance BagOfWords."""
        return BagOfWords(max_features=100)

    @pytest.fixture
    def sample_docs(self):
        """Documents d'exemple."""
        return [
            ["hello", "world", "test"],
            ["hello", "python", "code"],
            ["world", "data", "test"],
        ]

    def test_fit_transform(self, bow, sample_docs):
        """Test fit_transform."""
        matrix = bow.fit_transform(sample_docs)

        assert matrix is not None
        assert matrix.shape[0] == len(sample_docs)

    def test_word_counts(self, bow):
        """Test comptage des mots."""
        docs = [["word", "word", "other"]]
        bow.fit(docs)
        matrix = bow.transform(docs)

        # "word" appears twice
        assert np.max(matrix) >= 2


class TestNGramExtractor:
    """Tests pour NGramExtractor."""

    def test_unigrams(self):
        """Test extraction unigrams."""
        extractor = NGramExtractor(n=1)
        tokens = ["hello", "world", "test"]
        ngrams = extractor.extract(tokens)

        assert len(ngrams) == 3
        assert "hello" in ngrams

    def test_bigrams(self):
        """Test extraction bigrams."""
        extractor = NGramExtractor(n=2)
        tokens = ["hello", "world", "test"]
        ngrams = extractor.extract(tokens)

        assert len(ngrams) == 2
        assert "hello_world" in ngrams
        assert "world_test" in ngrams

    def test_trigrams(self):
        """Test extraction trigrams."""
        extractor = NGramExtractor(n=3)
        tokens = ["a", "b", "c", "d"]
        ngrams = extractor.extract(tokens)

        assert len(ngrams) == 2
        assert "a_b_c" in ngrams


class TestFeatureExtractor:
    """Tests pour FeatureExtractor."""

    @pytest.fixture
    def sample_docs(self):
        """Documents d'exemple."""
        return [
            ["python", "machine", "learning"],
            ["java", "software", "development"],
            ["python", "data", "science"],
        ]

    def test_tfidf_method(self, sample_docs):
        """Test avec methode tfidf."""
        extractor = FeatureExtractor(method='tfidf')
        matrix = extractor.fit_transform(sample_docs)

        assert matrix is not None
        assert matrix.shape[0] == len(sample_docs)

    def test_bow_method(self, sample_docs):
        """Test avec methode bow."""
        extractor = FeatureExtractor(method='bow')
        matrix = extractor.fit_transform(sample_docs)

        assert matrix is not None
        assert matrix.shape[0] == len(sample_docs)

    def test_ngram_range(self, sample_docs):
        """Test avec n-grams."""
        extractor = FeatureExtractor(method='tfidf', ngram_range=(1, 2))
        matrix = extractor.fit_transform(sample_docs)

        # Should have more features with bigrams
        assert matrix.shape[1] > 0
