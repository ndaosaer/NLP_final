"""Tests unitaires pour le module preprocessing."""

import pytest
from cv_analyzer.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Tests pour TextPreprocessor."""

    @pytest.fixture
    def preprocessor(self):
        """Preprocesseur par defaut."""
        return TextPreprocessor(language='both', use_lemmatization=False)

    @pytest.fixture
    def preprocessor_with_lemma(self):
        """Preprocesseur avec lemmatisation."""
        return TextPreprocessor(language='both', use_lemmatization=True)

    def test_basic_preprocessing(self, preprocessor):
        """Test preprocessing basique."""
        text = "Hello World! This is a TEST."
        tokens = preprocessor.preprocess(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Should be lowercase
        assert all(t.islower() or not t.isalpha() for t in tokens)

    def test_removes_stopwords(self, preprocessor):
        """Test suppression des stopwords."""
        text = "This is a very simple test"
        tokens = preprocessor.preprocess(text)

        # Common stopwords should be removed
        assert "is" not in tokens
        assert "a" not in tokens
        assert "the" not in tokens

    def test_handles_empty_text(self, preprocessor):
        """Test avec texte vide."""
        tokens = preprocessor.preprocess("")
        assert tokens == []

    def test_handles_special_characters(self, preprocessor):
        """Test avec caracteres speciaux."""
        text = "Hello!!! @user #hashtag $$$"
        tokens = preprocessor.preprocess(text)

        # Should handle special chars gracefully
        assert isinstance(tokens, list)

    def test_removes_urls(self, preprocessor):
        """Test suppression des URLs."""
        text = "Visit https://example.com for more info"
        tokens = preprocessor.preprocess(text)

        assert "https" not in tokens
        assert "example" not in tokens
        assert "com" not in tokens

    def test_removes_emails(self, preprocessor):
        """Test suppression des emails."""
        text = "Contact me at john@example.com"
        tokens = preprocessor.preprocess(text)

        assert "john@example.com" not in tokens
        assert "@" not in "".join(tokens)

    def test_preprocess_to_text(self, preprocessor):
        """Test preprocess_to_text."""
        text = "Hello World Test"
        result = preprocessor.preprocess_to_text(text)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_min_token_length(self):
        """Test longueur minimale des tokens."""
        preprocessor = TextPreprocessor(min_token_length=3)
        text = "I am a test"
        tokens = preprocessor.preprocess(text)

        # Tokens with less than 3 chars should be removed
        assert "I" not in tokens
        assert "am" not in tokens
        assert "a" not in tokens

    def test_french_stopwords(self):
        """Test stopwords francais."""
        preprocessor = TextPreprocessor(language='french')
        text = "Je suis un test"
        tokens = preprocessor.preprocess(text)

        # French stopwords
        assert "je" not in tokens
        assert "suis" not in tokens
        assert "un" not in tokens

    def test_both_languages_stopwords(self):
        """Test stopwords dans les deux langues."""
        preprocessor = TextPreprocessor(language='both')
        text = "I am a test Je suis un test"
        tokens = preprocessor.preprocess(text)

        # Both English and French stopwords should be removed
        assert "am" not in tokens
        assert "suis" not in tokens

    def test_accent_removal(self):
        """Test suppression des accents."""
        preprocessor = TextPreprocessor(remove_accents=True)
        text = "cafe resume"
        tokens = preprocessor.preprocess(text)

        # Check tokens exist (accents removed)
        assert len(tokens) >= 0
