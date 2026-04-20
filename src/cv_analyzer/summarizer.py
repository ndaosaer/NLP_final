"""
Module de résumé extractif pour les CV
Techniques classiques : TF-IDF scoring, TextRank, fréquence de mots
"""

import re
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter

from .preprocessing import TextPreprocessor


class SentenceCleaner:
    """
    Nettoyeur de phrases pour le résumé.
    Applique plusieurs passes de nettoyage pour un résultat propre.
    """

    # Patterns pour les données à supprimer
    EMAIL_PATTERN = re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b')
    PHONE_PATTERN = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}')
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

    # Caractères résiduels à nettoyer
    RESIDUAL_CHARS = re.compile(r'^[\s\-\*\•\·\>\|\:]+|[\s\-\*\•\·\>\|\:]+$')
    EMPTY_PARENS = re.compile(r'\(\s*\)|\[\s*\]|\{\s*\}')
    ISOLATED_DATES = re.compile(r'^\s*\(?\d{4}\s*[-–]\s*\d{4}\)?\s*$')

    @classmethod
    def clean_sentence(cls, sentence: str) -> str:
        """
        Nettoie une phrase pour le résumé.

        Args:
            sentence: Phrase brute

        Returns:
            Phrase nettoyée
        """
        if not sentence:
            return ""

        text = sentence

        # 1. Supprimer emails, téléphones, URLs
        text = cls.EMAIL_PATTERN.sub('', text)
        text = cls.PHONE_PATTERN.sub('', text)
        text = cls.URL_PATTERN.sub('', text)

        # 2. Supprimer parenthèses/crochets vides
        text = cls.EMPTY_PARENS.sub('', text)

        # 3. Supprimer caractères résiduels en début/fin
        text = cls.RESIDUAL_CHARS.sub('', text)

        # 4. Normaliser les espaces autour de la ponctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Pas d'espace avant ponctuation
        text = re.sub(r'([.,;:!?])(?=[^\s])', r'\1 ', text)  # Espace après ponctuation

        # 5. Normaliser les tirets et apostrophes
        text = re.sub(r'\s*[-–—]\s*', ' - ', text)
        text = re.sub(r"['']", "'", text)

        # 6. Supprimer espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()

        # 7. Capitaliser la première lettre si nécessaire
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # 8. S'assurer que la phrase se termine par une ponctuation
        if text and text[-1] not in '.!?':
            text += '.'

        return text

    @classmethod
    def clean_sentences(cls, sentences: list) -> list:
        """
        Nettoie une liste de phrases.

        Args:
            sentences: Liste de phrases brutes

        Returns:
            Liste de phrases nettoyées (phrases vides filtrées)
        """
        cleaned = []
        for sent in sentences:
            clean = cls.clean_sentence(sent)
            # Filtrer les phrases trop courtes après nettoyage
            if len(clean) >= 15:
                cleaned.append(clean)
        return cleaned


class SentenceTokenizer:
    """
    Tokeniseur de phrases simple.
    Découpe un texte en phrases individuelles.
    """

    # Patterns pour détecter les fins de phrases
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    # Patterns pour les sections de CV
    CV_SECTIONS = [
        'experience', 'education', 'skills', 'competences',
        'formation', 'diplomes', 'certifications', 'languages',
        'langues', 'projets', 'projects', 'references', 'contact',
        'profil', 'profile', 'summary', 'objective', 'objectif'
    ]

    def tokenize(self, text: str) -> List[str]:
        """
        Découpe le texte en phrases.

        Args:
            text: Texte brut

        Returns:
            Liste de phrases
        """
        # Nettoyer le texte
        text = self._clean_text(text)

        # Découper par retours à la ligne d'abord (structure CV)
        lines = text.split('\n')

        sentences = []
        for line in lines:
            line = line.strip()
            if len(line) < 10:  # Ignorer les lignes trop courtes
                continue

            # Découper par ponctuation
            parts = self.SENTENCE_ENDINGS.split(line)
            for part in parts:
                part = part.strip()
                if len(part) >= 15:  # Phrase minimale
                    sentences.append(part)

        return sentences

    def _clean_text(self, text: str) -> str:
        """Nettoie le texte pour la tokenisation."""
        # Supprimer les caractères spéciaux excessifs
        text = re.sub(r'[•●◦▪■□►▸→–—]', ' ', text)
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        return text


class TFIDFSummarizer:
    """
    Résumé extractif basé sur les scores TF-IDF.

    Principe: Les phrases contenant les mots les plus importants
    (haut score TF-IDF) sont sélectionnées pour le résumé.
    """

    def __init__(self, language: str = 'both'):
        """
        Initialise le summarizer TF-IDF.

        Args:
            language: Langue pour les stopwords
        """
        self.preprocessor = TextPreprocessor(
            language=language,
            remove_accents=True,
            use_lemmatization=False  # Garder les mots originaux
        )
        self.sentence_tokenizer = SentenceTokenizer()

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calcule les fréquences de termes."""
        counter = Counter(tokens)
        total = len(tokens)
        return {word: count / total for word, count in counter.items()}

    def _compute_idf(self, sentences_tokens: List[List[str]]) -> Dict[str, float]:
        """Calcule les IDF pour tous les mots."""
        n_docs = len(sentences_tokens)
        doc_freq = Counter()

        for tokens in sentences_tokens:
            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)

        idf = {}
        for word, freq in doc_freq.items():
            idf[word] = math.log((n_docs + 1) / (freq + 1)) + 1

        return idf

    def _score_sentence(
        self,
        sentence_tokens: List[str],
        idf: Dict[str, float]
    ) -> float:
        """
        Calcule le score d'une phrase basé sur TF-IDF.

        Args:
            sentence_tokens: Tokens de la phrase
            idf: Dictionnaire IDF

        Returns:
            Score de la phrase
        """
        if not sentence_tokens:
            return 0.0

        tf = self._compute_tf(sentence_tokens)
        score = sum(tf.get(word, 0) * idf.get(word, 0) for word in sentence_tokens)

        # Normaliser par la longueur
        return score / len(sentence_tokens)

    def summarize(
        self,
        text: str,
        num_sentences: int = 3,
        min_sentence_length: int = 20
    ) -> List[str]:
        """
        Génère un résumé extractif.

        Args:
            text: Texte du CV
            num_sentences: Nombre de phrases à extraire
            min_sentence_length: Longueur minimale des phrases

        Returns:
            Liste des phrases les plus importantes
        """
        # Tokeniser en phrases
        sentences = self.sentence_tokenizer.tokenize(text)

        if len(sentences) <= num_sentences:
            return sentences

        # Prétraiter chaque phrase
        sentences_tokens = [
            self.preprocessor.preprocess(sent)
            for sent in sentences
        ]

        # Calculer IDF sur tout le document
        idf = self._compute_idf(sentences_tokens)

        # Scorer chaque phrase
        scored_sentences = []
        for i, (sentence, tokens) in enumerate(zip(sentences, sentences_tokens)):
            if len(sentence) >= min_sentence_length and len(tokens) > 2:
                score = self._score_sentence(tokens, idf)
                scored_sentences.append((i, sentence, score))

        # Trier par score
        scored_sentences.sort(key=lambda x: x[2], reverse=True)

        # Sélectionner les top phrases
        top_sentences = scored_sentences[:num_sentences]

        # Remettre dans l'ordre original
        top_sentences.sort(key=lambda x: x[0])

        return [sent for _, sent, _ in top_sentences]


class TextRankSummarizer:
    """
    Résumé extractif basé sur l'algorithme TextRank.

    Principe: Algorithme de graphe inspiré de PageRank.
    Les phrases sont des noeuds, les arêtes sont les similarités.
    Les phrases avec le plus de connexions importantes sont sélectionnées.
    """

    def __init__(
        self,
        language: str = 'both',
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-4
    ):
        """
        Initialise le summarizer TextRank.

        Args:
            language: Langue pour les stopwords
            damping: Facteur d'amortissement (comme PageRank)
            max_iter: Nombre maximum d'itérations
            tol: Tolérance pour la convergence
        """
        self.preprocessor = TextPreprocessor(
            language=language,
            remove_accents=True,
            use_lemmatization=True
        )
        self.sentence_tokenizer = SentenceTokenizer()
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def _cosine_similarity(
        self,
        tokens1: List[str],
        tokens2: List[str]
    ) -> float:
        """
        Calcule la similarité cosinus entre deux ensembles de tokens.

        Args:
            tokens1: Tokens de la première phrase
            tokens2: Tokens de la deuxième phrase

        Returns:
            Score de similarité [0, 1]
        """
        set1, set2 = set(tokens1), set(tokens2)

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        norm = math.sqrt(len(set1)) * math.sqrt(len(set2))

        return intersection / norm if norm > 0 else 0.0

    def _build_similarity_matrix(
        self,
        sentences_tokens: List[List[str]]
    ) -> np.ndarray:
        """
        Construit la matrice de similarité entre phrases.

        Args:
            sentences_tokens: Liste des phrases tokenisées

        Returns:
            Matrice de similarité
        """
        n = len(sentences_tokens)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self._cosine_similarity(
                        sentences_tokens[i],
                        sentences_tokens[j]
                    )

        # Normaliser les lignes
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Éviter division par zéro
        matrix = matrix / row_sums

        return matrix

    def _textrank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Applique l'algorithme TextRank.

        Args:
            similarity_matrix: Matrice de similarité

        Returns:
            Scores TextRank pour chaque phrase
        """
        n = similarity_matrix.shape[0]

        # Initialiser les scores uniformément
        scores = np.ones(n) / n

        for _ in range(self.max_iter):
            prev_scores = scores.copy()

            # Mise à jour TextRank
            scores = (1 - self.damping) / n + self.damping * similarity_matrix.T @ scores

            # Vérifier la convergence
            if np.abs(scores - prev_scores).sum() < self.tol:
                break

        return scores

    def summarize(
        self,
        text: str,
        num_sentences: int = 3,
        min_sentence_length: int = 20
    ) -> List[str]:
        """
        Génère un résumé extractif avec TextRank.

        Args:
            text: Texte du CV
            num_sentences: Nombre de phrases à extraire
            min_sentence_length: Longueur minimale des phrases

        Returns:
            Liste des phrases les plus importantes
        """
        # Tokeniser en phrases
        sentences = self.sentence_tokenizer.tokenize(text)

        # Filtrer les phrases trop courtes
        valid_sentences = [
            (i, sent) for i, sent in enumerate(sentences)
            if len(sent) >= min_sentence_length
        ]

        if len(valid_sentences) <= num_sentences:
            return [sent for _, sent in valid_sentences]

        # Prétraiter chaque phrase
        sentences_tokens = [
            self.preprocessor.preprocess(sent)
            for _, sent in valid_sentences
        ]

        # Filtrer les phrases avec trop peu de tokens
        filtered = [
            (idx, sent, tokens)
            for (idx, sent), tokens in zip(valid_sentences, sentences_tokens)
            if len(tokens) > 2
        ]

        if len(filtered) <= num_sentences:
            return [sent for _, sent, _ in filtered]

        # Construire la matrice de similarité
        tokens_only = [tokens for _, _, tokens in filtered]
        similarity_matrix = self._build_similarity_matrix(tokens_only)

        # Appliquer TextRank
        scores = self._textrank(similarity_matrix)

        # Associer les scores aux phrases
        scored = [
            (orig_idx, sent, score)
            for (orig_idx, sent, _), score in zip(filtered, scores)
        ]

        # Trier par score
        scored.sort(key=lambda x: x[2], reverse=True)

        # Sélectionner les top phrases
        top = scored[:num_sentences]

        # Remettre dans l'ordre original
        top.sort(key=lambda x: x[0])

        return [sent for _, sent, _ in top]


class FrequencySummarizer:
    """
    Résumé extractif basé sur la fréquence des mots.

    Principe simple: Les phrases contenant les mots les plus fréquents
    sont considérées comme les plus importantes.
    """

    def __init__(self, language: str = 'both'):
        """
        Initialise le summarizer par fréquence.

        Args:
            language: Langue pour les stopwords
        """
        self.preprocessor = TextPreprocessor(
            language=language,
            remove_accents=True,
            use_lemmatization=False
        )
        self.sentence_tokenizer = SentenceTokenizer()

    def summarize(
        self,
        text: str,
        num_sentences: int = 3,
        min_sentence_length: int = 20
    ) -> List[str]:
        """
        Génère un résumé basé sur la fréquence des mots.

        Args:
            text: Texte du CV
            num_sentences: Nombre de phrases à extraire
            min_sentence_length: Longueur minimale des phrases

        Returns:
            Liste des phrases les plus importantes
        """
        # Tokeniser en phrases
        sentences = self.sentence_tokenizer.tokenize(text)

        if len(sentences) <= num_sentences:
            return sentences

        # Prétraiter tout le texte pour obtenir les fréquences globales
        all_tokens = self.preprocessor.preprocess(text)
        word_freq = Counter(all_tokens)

        # Normaliser les fréquences
        max_freq = max(word_freq.values()) if word_freq else 1
        word_freq = {word: freq / max_freq for word, freq in word_freq.items()}

        # Scorer chaque phrase
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            if len(sentence) < min_sentence_length:
                continue

            tokens = self.preprocessor.preprocess(sentence)
            if len(tokens) < 3:
                continue

            # Score = somme des fréquences normalisées
            score = sum(word_freq.get(token, 0) for token in tokens)
            score = score / len(tokens)  # Normaliser par longueur

            scored_sentences.append((i, sentence, score))

        # Trier par score
        scored_sentences.sort(key=lambda x: x[2], reverse=True)

        # Sélectionner les top phrases
        top = scored_sentences[:num_sentences]

        # Remettre dans l'ordre original
        top.sort(key=lambda x: x[0])

        return [sent for _, sent, _ in top]


class CVSummarizer:
    """
    Classe principale pour le résumé de CV.
    Combine plusieurs méthodes et permet de choisir l'algorithme.
    """

    METHODS = {
        'tfidf': TFIDFSummarizer,
        'textrank': TextRankSummarizer,
        'frequency': FrequencySummarizer
    }

    def __init__(
        self,
        method: str = 'tfidf',
        language: str = 'both'
    ):
        """
        Initialise le summarizer de CV.

        Args:
            method: Méthode de résumé ('tfidf', 'textrank', 'frequency')
            language: Langue pour les stopwords
        """
        if method not in self.METHODS:
            raise ValueError(
                f"Methode inconnue: {method}. "
                f"Disponibles: {list(self.METHODS.keys())}"
            )

        self.method = method
        self.summarizer = self.METHODS[method](language=language)

    def summarize(
        self,
        text: str,
        num_sentences: int = 5,
        min_sentence_length: int = 20
    ) -> str:
        """
        Génère un résumé du CV.

        Args:
            text: Texte du CV
            num_sentences: Nombre de phrases à extraire
            min_sentence_length: Longueur minimale des phrases

        Returns:
            Résumé sous forme de texte
        """
        sentences = self.summarizer.summarize(
            text,
            num_sentences=num_sentences,
            min_sentence_length=min_sentence_length
        )

        # Appliquer le nettoyage post-traitement
        cleaned_sentences = SentenceCleaner.clean_sentences(sentences)

        return '\n'.join(f"- {sent}" for sent in cleaned_sentences)

    def summarize_as_list(
        self,
        text: str,
        num_sentences: int = 5,
        min_sentence_length: int = 20
    ) -> List[str]:
        """
        Génère un résumé sous forme de liste.

        Args:
            text: Texte du CV
            num_sentences: Nombre de phrases à extraire
            min_sentence_length: Longueur minimale des phrases

        Returns:
            Liste des phrases du résumé
        """
        sentences = self.summarizer.summarize(
            text,
            num_sentences=num_sentences,
            min_sentence_length=min_sentence_length
        )
        return SentenceCleaner.clean_sentences(sentences)


# Fonction utilitaire
def summarize_cv(
    text: str,
    method: str = 'tfidf',
    num_sentences: int = 5
) -> str:
    """
    Fonction utilitaire pour résumer un CV.

    Args:
        text: Texte du CV
        method: Méthode ('tfidf', 'textrank', 'frequency')
        num_sentences: Nombre de phrases

    Returns:
        Résumé
    """
    summarizer = CVSummarizer(method=method)
    return summarizer.summarize(text, num_sentences=num_sentences)


# Test du module
if __name__ == "__main__":
    sample_cv = """
    Jean DUPONT - Data Scientist Senior

    PROFIL
    Data Scientist passionné avec 5 ans d'expérience dans le développement
    de solutions d'intelligence artificielle. Expert en machine learning
    et traitement du langage naturel. Capacité à transformer des données
    complexes en insights business actionnables.

    EXPERIENCE PROFESSIONNELLE

    Data Scientist Senior - Entreprise ABC (2021-2024)
    Développement de modèles de machine learning pour la prédiction
    des ventes avec une amélioration de 25% de la précision.
    Mise en place de pipelines de données automatisés avec Apache Airflow.
    Collaboration avec les équipes produit pour intégrer les modèles en production.

    Data Scientist - Startup XYZ (2019-2021)
    Création d'un système de recommandation personnalisé augmentant
    l'engagement utilisateur de 40%. Implémentation de modèles NLP
    pour l'analyse de sentiment des avis clients.

    FORMATION
    Master en Intelligence Artificielle - Université Paris-Saclay (2019)
    Licence en Mathématiques Appliquées - Université Lyon 1 (2017)

    COMPETENCES TECHNIQUES
    Python, R, SQL, TensorFlow, PyTorch, Scikit-learn, Pandas
    Machine Learning, Deep Learning, NLP, Computer Vision
    Docker, Kubernetes, AWS, GCP, Apache Spark
    """

    print("=" * 60)
    print("TEST DES METHODES DE RESUME")
    print("=" * 60)

    for method in ['tfidf', 'textrank', 'frequency']:
        print(f"\n--- Methode: {method.upper()} ---\n")
        summarizer = CVSummarizer(method=method)
        summary = summarizer.summarize(sample_cv, num_sentences=3)
        print(summary)
