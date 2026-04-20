"""Tests unitaires pour le module summarizer."""

import pytest
from cv_analyzer.summarizer import (
    CVSummarizer,
    TFIDFSummarizer,
    TextRankSummarizer,
    FrequencySummarizer,
    SentenceTokenizer,
    SentenceCleaner,
)


class TestSentenceTokenizer:
    """Tests pour SentenceTokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Instance de tokenizer."""
        return SentenceTokenizer()

    def test_basic_tokenization(self, tokenizer):
        """Test tokenization basique."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = tokenizer.tokenize(text)

        assert len(sentences) >= 1

    def test_handles_empty_text(self, tokenizer):
        """Test avec texte vide."""
        sentences = tokenizer.tokenize("")
        assert sentences == []

    def test_handles_short_lines(self, tokenizer):
        """Test avec lignes courtes."""
        text = "Hi\nWorld\nThis is a longer sentence that should be included."
        sentences = tokenizer.tokenize(text)

        # Short lines should be filtered
        assert all(len(s) >= 10 for s in sentences)


class TestSentenceCleaner:
    """Tests pour SentenceCleaner."""

    def test_clean_sentence_basic(self):
        """Test nettoyage basique."""
        sentence = "  hello world  "
        cleaned = SentenceCleaner.clean_sentence(sentence)

        assert cleaned.strip() == cleaned
        assert cleaned[0].isupper()

    def test_removes_email(self):
        """Test suppression email."""
        sentence = "Contact me at john@example.com for details"
        cleaned = SentenceCleaner.clean_sentence(sentence)

        assert "@" not in cleaned
        assert "john@example.com" not in cleaned

    def test_removes_url(self):
        """Test suppression URL."""
        sentence = "Visit https://example.com for more"
        cleaned = SentenceCleaner.clean_sentence(sentence)

        assert "https://" not in cleaned
        assert "example.com" not in cleaned

    def test_adds_punctuation(self):
        """Test ajout ponctuation."""
        sentence = "This is a sentence without period"
        cleaned = SentenceCleaner.clean_sentence(sentence)

        assert cleaned.endswith('.')

    def test_capitalizes_first_letter(self):
        """Test majuscule premiere lettre."""
        sentence = "lowercase start"
        cleaned = SentenceCleaner.clean_sentence(sentence)

        assert cleaned[0].isupper()

    def test_clean_sentences_list(self):
        """Test nettoyage liste de phrases."""
        sentences = [
            "First sentence here",
            "Short",  # Too short, should be filtered
            "Second sentence that is long enough to be included",
        ]
        cleaned = SentenceCleaner.clean_sentences(sentences)

        assert len(cleaned) == 2


class TestTFIDFSummarizer:
    """Tests pour TFIDFSummarizer."""

    @pytest.fixture
    def summarizer(self):
        """Instance de summarizer."""
        return TFIDFSummarizer()

    def test_summarize_basic(self, summarizer, sample_cv_data_scientist):
        """Test resume basique."""
        sentences = summarizer.summarize(sample_cv_data_scientist, num_sentences=3)

        assert isinstance(sentences, list)
        assert len(sentences) <= 3
        assert all(isinstance(s, str) for s in sentences)

    def test_summarize_respects_num_sentences(self, summarizer, sample_cv_data_scientist):
        """Test respect du nombre de phrases."""
        for n in [2, 3, 5]:
            sentences = summarizer.summarize(sample_cv_data_scientist, num_sentences=n)
            assert len(sentences) <= n

    def test_summarize_short_text(self, summarizer):
        """Test avec texte court."""
        text = "This is a short text. It has only two sentences."
        sentences = summarizer.summarize(text, num_sentences=5)

        # Should return at most the available sentences
        assert len(sentences) <= 2


class TestTextRankSummarizer:
    """Tests pour TextRankSummarizer."""

    @pytest.fixture
    def summarizer(self):
        """Instance de summarizer."""
        return TextRankSummarizer()

    def test_summarize_basic(self, summarizer, sample_cv_data_scientist):
        """Test resume basique."""
        sentences = summarizer.summarize(sample_cv_data_scientist, num_sentences=3)

        assert isinstance(sentences, list)
        assert len(sentences) <= 3

    def test_summarize_different_damping(self, sample_cv_data_scientist):
        """Test avec different facteur d'amortissement."""
        summarizer = TextRankSummarizer(damping=0.5)
        sentences = summarizer.summarize(sample_cv_data_scientist, num_sentences=3)

        assert len(sentences) <= 3


class TestFrequencySummarizer:
    """Tests pour FrequencySummarizer."""

    @pytest.fixture
    def summarizer(self):
        """Instance de summarizer."""
        return FrequencySummarizer()

    def test_summarize_basic(self, summarizer, sample_cv_data_scientist):
        """Test resume basique."""
        sentences = summarizer.summarize(sample_cv_data_scientist, num_sentences=3)

        assert isinstance(sentences, list)
        assert len(sentences) <= 3


class TestCVSummarizer:
    """Tests pour CVSummarizer (classe principale)."""

    def test_available_methods(self):
        """Test methodes disponibles."""
        assert 'tfidf' in CVSummarizer.METHODS
        assert 'textrank' in CVSummarizer.METHODS
        assert 'frequency' in CVSummarizer.METHODS

    def test_invalid_method(self):
        """Test methode invalide."""
        with pytest.raises(ValueError):
            CVSummarizer(method='invalid')

    @pytest.mark.parametrize("method", ["tfidf", "textrank", "frequency"])
    def test_all_methods(self, method, sample_cv_data_scientist):
        """Test toutes les methodes."""
        summarizer = CVSummarizer(method=method)
        result = summarizer.summarize(sample_cv_data_scientist, num_sentences=3)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_as_list(self, sample_cv_data_scientist):
        """Test summarize_as_list."""
        summarizer = CVSummarizer(method='tfidf')
        sentences = summarizer.summarize_as_list(sample_cv_data_scientist, num_sentences=3)

        assert isinstance(sentences, list)
        assert all(isinstance(s, str) for s in sentences)

    def test_summarize_format(self, sample_cv_data_scientist):
        """Test format du resume."""
        summarizer = CVSummarizer(method='tfidf')
        result = summarizer.summarize(sample_cv_data_scientist, num_sentences=3)

        # Should be bullet-point format
        assert "- " in result
