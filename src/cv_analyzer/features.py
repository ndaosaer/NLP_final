"""
Module d'extraction de features pour les CV
Techniques : TF-IDF, Bag of Words, N-grams
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from collections import Counter
import math


class BagOfWords:
    """
    Implémentation manuelle de Bag of Words (sac de mots).

    Crée une représentation vectorielle basée sur le comptage des mots.
    """

    def __init__(self, max_features: Optional[int] = None, min_df: int = 1):
        """
        Initialise le vectoriseur Bag of Words.

        Args:
            max_features: Nombre maximum de features à conserver (les plus fréquents)
            min_df: Fréquence documentaire minimale (nombre de documents)
        """
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary: Dict[str, int] = {}
        self.feature_names: List[str] = []
        self._is_fitted = False

    def fit(self, documents: List[List[str]]) -> 'BagOfWords':
        """
        Apprend le vocabulaire à partir des documents.

        Args:
            documents: Liste de documents (chaque document est une liste de tokens)

        Returns:
            self
        """
        # Compter la fréquence documentaire de chaque mot
        doc_frequency = Counter()
        word_frequency = Counter()

        for doc in documents:
            unique_words = set(doc)
            doc_frequency.update(unique_words)
            word_frequency.update(doc)

        # Filtrer par fréquence documentaire minimale
        valid_words = {
            word for word, freq in doc_frequency.items()
            if freq >= self.min_df
        }

        # Trier par fréquence totale (décroissant)
        sorted_words = sorted(
            valid_words,
            key=lambda w: word_frequency[w],
            reverse=True
        )

        # Limiter le nombre de features
        if self.max_features:
            sorted_words = sorted_words[:self.max_features]

        # Créer le vocabulaire (mot -> index)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted_words)}
        self.feature_names = sorted_words
        self._is_fitted = True

        return self

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """
        Transforme les documents en vecteurs BoW.

        Args:
            documents: Liste de documents tokenisés

        Returns:
            Matrice numpy (n_documents, n_features)
        """
        if not self._is_fitted:
            raise ValueError("Le vectoriseur doit être fit avant transform")

        n_docs = len(documents)
        n_features = len(self.vocabulary)

        # Créer la matrice de comptage
        matrix = np.zeros((n_docs, n_features), dtype=np.float64)

        for doc_idx, doc in enumerate(documents):
            word_counts = Counter(doc)
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    feature_idx = self.vocabulary[word]
                    matrix[doc_idx, feature_idx] = count

        return matrix

    def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
        """
        Fit et transform en une seule opération.

        Args:
            documents: Liste de documents tokenisés

        Returns:
            Matrice BoW
        """
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self) -> List[str]:
        """Retourne la liste des features (mots du vocabulaire)."""
        return self.feature_names


class TFIDF:
    """
    Implémentation manuelle de TF-IDF (Term Frequency - Inverse Document Frequency).

    TF(t,d) = nombre d'occurrences de t dans d / nombre total de termes dans d
    IDF(t) = log(N / df(t)) + 1
    TF-IDF(t,d) = TF(t,d) * IDF(t)
    """

    def __init__(
        self,
        max_features: Optional[int] = None,
        min_df: int = 1,
        max_df: float = 1.0,
        norm: str = 'l2',
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False
    ):
        """
        Initialise le vectoriseur TF-IDF.

        Args:
            max_features: Nombre maximum de features
            min_df: Fréquence documentaire minimale (int: count, float: proportion)
            max_df: Fréquence documentaire maximale (float: proportion)
            norm: Normalisation ('l1', 'l2', ou None)
            use_idf: Utiliser IDF
            smooth_idf: Ajouter 1 au dénominateur IDF pour éviter division par 0
            sublinear_tf: Utiliser 1 + log(TF) au lieu de TF
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

        self.vocabulary: Dict[str, int] = {}
        self.feature_names: List[str] = []
        self.idf_values: np.ndarray = np.array([])
        self._n_documents = 0
        self._is_fitted = False

    def fit(self, documents: List[List[str]]) -> 'TFIDF':
        """
        Apprend le vocabulaire et calcule les IDF.

        Args:
            documents: Liste de documents tokenisés

        Returns:
            self
        """
        self._n_documents = len(documents)

        # Compter les fréquences documentaires
        doc_frequency = Counter()
        word_frequency = Counter()

        for doc in documents:
            unique_words = set(doc)
            doc_frequency.update(unique_words)
            word_frequency.update(doc)

        # Calculer les seuils min_df et max_df
        if isinstance(self.min_df, float):
            min_df_count = int(self.min_df * self._n_documents)
        else:
            min_df_count = self.min_df

        max_df_count = int(self.max_df * self._n_documents)

        # Filtrer les mots par fréquence documentaire
        valid_words = {
            word for word, freq in doc_frequency.items()
            if min_df_count <= freq <= max_df_count
        }

        # Trier par fréquence totale
        sorted_words = sorted(
            valid_words,
            key=lambda w: word_frequency[w],
            reverse=True
        )

        # Limiter le nombre de features
        if self.max_features:
            sorted_words = sorted_words[:self.max_features]

        # Créer le vocabulaire
        self.vocabulary = {word: idx for idx, word in enumerate(sorted_words)}
        self.feature_names = sorted_words

        # Calculer les IDF
        if self.use_idf:
            self.idf_values = np.zeros(len(self.vocabulary))

            for word, idx in self.vocabulary.items():
                df = doc_frequency[word]

                if self.smooth_idf:
                    # IDF avec lissage : log((N + 1) / (df + 1)) + 1
                    idf = math.log((self._n_documents + 1) / (df + 1)) + 1
                else:
                    # IDF standard : log(N / df) + 1
                    idf = math.log(self._n_documents / df) + 1

                self.idf_values[idx] = idf

        self._is_fitted = True
        return self

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """
        Transforme les documents en vecteurs TF-IDF.

        Args:
            documents: Liste de documents tokenisés

        Returns:
            Matrice TF-IDF (n_documents, n_features)
        """
        if not self._is_fitted:
            raise ValueError("Le vectoriseur doit être fit avant transform")

        n_docs = len(documents)
        n_features = len(self.vocabulary)

        matrix = np.zeros((n_docs, n_features), dtype=np.float64)

        for doc_idx, doc in enumerate(documents):
            # Calculer TF
            word_counts = Counter(doc)
            doc_length = len(doc)

            if doc_length == 0:
                continue

            for word, count in word_counts.items():
                if word in self.vocabulary:
                    feature_idx = self.vocabulary[word]

                    # Term Frequency
                    if self.sublinear_tf:
                        tf = 1 + math.log(count) if count > 0 else 0
                    else:
                        tf = count / doc_length

                    # TF * IDF
                    if self.use_idf:
                        matrix[doc_idx, feature_idx] = tf * self.idf_values[feature_idx]
                    else:
                        matrix[doc_idx, feature_idx] = tf

        # Normalisation
        if self.norm:
            matrix = self._normalize(matrix)

        return matrix

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        """Normalise les vecteurs."""
        if self.norm == 'l2':
            # Norme L2 (euclidienne)
            norms = np.sqrt((matrix ** 2).sum(axis=1, keepdims=True))
            norms[norms == 0] = 1  # Éviter division par 0
            return matrix / norms
        elif self.norm == 'l1':
            # Norme L1 (somme des valeurs absolues)
            norms = np.abs(matrix).sum(axis=1, keepdims=True)
            norms[norms == 0] = 1
            return matrix / norms
        return matrix

    def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
        """Fit et transform en une seule opération."""
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self) -> List[str]:
        """Retourne la liste des features."""
        return self.feature_names

    def get_idf_scores(self) -> Dict[str, float]:
        """Retourne les scores IDF pour chaque mot."""
        return {
            word: self.idf_values[idx]
            for word, idx in self.vocabulary.items()
        }


class NGramExtractor:
    """
    Extracteur de N-grams (bi-grams, tri-grams, etc.).
    """

    def __init__(self, n: int = 2):
        """
        Initialise l'extracteur.

        Args:
            n: Taille des n-grams (2 pour bi-grams, 3 pour tri-grams)
        """
        self.n = n

    def extract(self, tokens: List[str]) -> List[str]:
        """
        Extrait les n-grams d'une liste de tokens.

        Args:
            tokens: Liste de tokens

        Returns:
            Liste de n-grams (sous forme de strings)
        """
        if len(tokens) < self.n:
            return []

        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngram = '_'.join(tokens[i:i + self.n])
            ngrams.append(ngram)

        return ngrams

    def extract_range(self, tokens: List[str], min_n: int = 1, max_n: int = None) -> List[str]:
        """
        Extrait des n-grams de tailles variées.

        Args:
            tokens: Liste de tokens
            min_n: Taille minimale
            max_n: Taille maximale (défaut: self.n)

        Returns:
            Liste de tous les n-grams
        """
        if max_n is None:
            max_n = self.n

        all_ngrams = []

        for n in range(min_n, max_n + 1):
            extractor = NGramExtractor(n)
            all_ngrams.extend(extractor.extract(tokens))

        return all_ngrams


class FeatureExtractor:
    """
    Classe principale pour l'extraction de features.
    Combine TF-IDF, BoW et N-grams.
    """

    def __init__(
        self,
        method: str = 'tfidf',
        max_features: int = 1500,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """
        Initialise l'extracteur de features.

        Args:
            method: 'tfidf' ou 'bow'
            max_features: Nombre maximum de features
            ngram_range: Tuple (min_n, max_n) pour les n-grams
            min_df: Fréquence documentaire minimale
            max_df: Fréquence documentaire maximale
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

        self.ngram_extractor = NGramExtractor(n=ngram_range[1])

        if method == 'tfidf':
            self.vectorizer = TFIDF(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df
            )
        else:
            self.vectorizer = BagOfWords(
                max_features=max_features,
                min_df=min_df
            )

        self._is_fitted = False

    def _add_ngrams(self, documents: List[List[str]]) -> List[List[str]]:
        """Ajoute les n-grams aux documents."""
        if self.ngram_range == (1, 1):
            return documents

        enriched_docs = []
        for doc in documents:
            ngrams = self.ngram_extractor.extract_range(
                doc,
                min_n=self.ngram_range[0],
                max_n=self.ngram_range[1]
            )
            enriched_docs.append(doc + ngrams)

        return enriched_docs

    def fit(self, documents: List[List[str]]) -> 'FeatureExtractor':
        """
        Apprend les features à partir des documents.

        Args:
            documents: Liste de documents tokenisés
        """
        # Ajouter les n-grams
        enriched_docs = self._add_ngrams(documents)

        # Fit le vectoriseur
        self.vectorizer.fit(enriched_docs)
        self._is_fitted = True

        return self

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """
        Transforme les documents en vecteurs de features.

        Args:
            documents: Liste de documents tokenisés

        Returns:
            Matrice de features
        """
        enriched_docs = self._add_ngrams(documents)
        return self.vectorizer.transform(enriched_docs)

    def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
        """Fit et transform en une seule opération."""
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self) -> List[str]:
        """Retourne les noms des features."""
        return self.vectorizer.get_feature_names()

    def get_top_features(self, vector: np.ndarray, n: int = 10) -> List[Tuple[str, float]]:
        """
        Retourne les features les plus importantes pour un vecteur.

        Args:
            vector: Vecteur de features (1D)
            n: Nombre de features à retourner

        Returns:
            Liste de tuples (feature_name, score)
        """
        feature_names = self.get_feature_names()

        # Indices des top features
        top_indices = np.argsort(vector)[::-1][:n]

        return [
            (feature_names[idx], vector[idx])
            for idx in top_indices
            if vector[idx] > 0
        ]


# Fonctions utilitaires

def extract_tfidf(documents: List[List[str]], **kwargs) -> Tuple[np.ndarray, TFIDF]:
    """
    Fonction utilitaire pour extraire TF-IDF.

    Args:
        documents: Liste de documents tokenisés
        **kwargs: Arguments pour TFIDF

    Returns:
        Tuple (matrice TF-IDF, vectoriseur)
    """
    vectorizer = TFIDF(**kwargs)
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer


def extract_bow(documents: List[List[str]], **kwargs) -> Tuple[np.ndarray, BagOfWords]:
    """
    Fonction utilitaire pour extraire Bag of Words.

    Args:
        documents: Liste de documents tokenisés
        **kwargs: Arguments pour BagOfWords

    Returns:
        Tuple (matrice BoW, vectoriseur)
    """
    vectorizer = BagOfWords(**kwargs)
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer


# Test du module
if __name__ == "__main__":
    # Documents de test
    documents = [
        ['python', 'machine', 'learning', 'data', 'science', 'python'],
        ['java', 'spring', 'backend', 'api', 'database'],
        ['python', 'django', 'web', 'api', 'backend'],
        ['machine', 'learning', 'deep', 'learning', 'tensorflow', 'python'],
        ['javascript', 'react', 'frontend', 'web', 'css'],
    ]

    print("=== Test Bag of Words ===")
    bow = BagOfWords(max_features=10)
    bow_matrix = bow.fit_transform(documents)
    print(f"Shape: {bow_matrix.shape}")
    print(f"Vocabulaire: {bow.get_feature_names()[:10]}")
    print(f"Premier document: {bow_matrix[0]}")
    print()

    print("=== Test TF-IDF ===")
    tfidf = TFIDF(max_features=10)
    tfidf_matrix = tfidf.fit_transform(documents)
    print(f"Shape: {tfidf_matrix.shape}")
    print(f"Vocabulaire: {tfidf.get_feature_names()[:10]}")
    print(f"IDF scores: {list(tfidf.get_idf_scores().items())[:5]}")
    print(f"Premier document (TF-IDF): {np.round(tfidf_matrix[0], 3)}")
    print()

    print("=== Test N-grams ===")
    ngram = NGramExtractor(n=2)
    tokens = ['machine', 'learning', 'with', 'python']
    bigrams = ngram.extract(tokens)
    print(f"Tokens: {tokens}")
    print(f"Bi-grams: {bigrams}")
    print()

    print("=== Test FeatureExtractor complet ===")
    extractor = FeatureExtractor(
        method='tfidf',
        max_features=15,
        ngram_range=(1, 2),
        min_df=1
    )
    features = extractor.fit_transform(documents)
    print(f"Shape avec bi-grams: {features.shape}")
    print(f"Features: {extractor.get_feature_names()}")
    print()

    print("=== Top features du premier document ===")
    top = extractor.get_top_features(features[0], n=5)
    for feat, score in top:
        print(f"  {feat}: {score:.4f}")
