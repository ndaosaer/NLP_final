"""
Classificateur de CV avec techniques NLP classiques
Utilise notre pipeline de preprocessing et extraction de features
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

# Nos modules NLP
from .preprocessing import TextPreprocessor
from .features import TFIDF, FeatureExtractor

# Sklearn pour les algorithmes de classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score
)


class CVClassifier:
    """
    Classificateur de CV utilisant des techniques NLP classiques.

    Pipeline:
    1. Prétraitement (tokenisation, lemmatisation, stopwords)
    2. Extraction de features (TF-IDF manuel)
    3. Classification (Naive Bayes, SVM, Logistic Regression, Random Forest)
    """

    # Algorithmes disponibles
    AVAILABLE_MODELS = {
        'naive_bayes': MultinomialNB,
        'logistic_regression': LogisticRegression,
        'svm': LinearSVC,
        'random_forest': RandomForestClassifier
    }

    def __init__(
        self,
        model_type: str = 'naive_bayes',
        max_features: int = 1500,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        language: str = 'both',
        use_lemmatization: bool = True
    ):
        """
        Initialise le classificateur.

        Args:
            model_type: Type de modèle ('naive_bayes', 'logistic_regression', 'svm', 'random_forest')
            max_features: Nombre maximum de features TF-IDF
            ngram_range: Tuple (min_n, max_n) pour les n-grams
            min_df: Fréquence documentaire minimale
            language: Langue pour les stopwords ('french', 'english', 'both')
            use_lemmatization: Utiliser la lemmatisation
        """
        self.model_type = model_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df

        # Préprocesseur NLP
        self.preprocessor = TextPreprocessor(
            language=language,
            remove_accents=True,
            min_token_length=2,
            use_lemmatization=use_lemmatization
        )

        # Extracteur de features TF-IDF
        self.feature_extractor = FeatureExtractor(
            method='tfidf',
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df
        )

        # Classificateur
        self.classifier = self._create_classifier(model_type)

        # Mapping des catégories
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}

        self._is_trained = False

    def _create_classifier(self, model_type: str):
        """Crée une instance du classificateur selon le type."""
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Modèle inconnu: {model_type}. "
                f"Disponibles: {list(self.AVAILABLE_MODELS.keys())}"
            )

        model_class = self.AVAILABLE_MODELS[model_type]

        # Configuration spécifique par modèle
        if model_type == 'logistic_regression':
            return model_class(max_iter=1000, random_state=42)
        elif model_type == 'svm':
            return model_class(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            return model_class(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            return model_class()

    def _encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """Encode les labels textuels en indices."""
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        return np.array([self.label_to_idx[label] for label in labels])

    def _decode_labels(self, indices: np.ndarray) -> np.ndarray:
        """Décode les indices en labels textuels."""
        return np.array([self.idx_to_label[idx] for idx in indices])

    def preprocess_documents(self, texts: List[str]) -> List[List[str]]:
        """
        Prétraite une liste de textes.

        Args:
            texts: Liste de textes bruts

        Returns:
            Liste de documents tokenisés
        """
        documents = []
        for text in texts:
            tokens = self.preprocessor.preprocess(str(text))
            documents.append(tokens)
        return documents

    def train(
        self,
        csv_path: str,
        text_column: str = 'Resume',
        category_column: str = 'Category',
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Entraîne le classificateur sur un dataset CSV.

        Args:
            csv_path: Chemin vers le fichier CSV
            text_column: Nom de la colonne contenant le texte
            category_column: Nom de la colonne contenant les catégories
            test_size: Proportion des données de test

        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        print(f"Chargement de {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"{len(df)} CV charges")

        # Extraire les données
        texts = df[text_column].tolist()
        labels = df[category_column].values

        # Prétraitement
        print("\nPretraitement des textes...")
        documents = self.preprocess_documents(texts)
        print(f"Vocabulaire moyen par document: {np.mean([len(d) for d in documents]):.1f} tokens")

        # Encodage des labels
        y = self._encode_labels(labels)
        print(f"Categories: {len(self.label_to_idx)}")

        # Split train/test
        print(f"\nSplit train/test ({int((1-test_size)*100)}/{int(test_size*100)})...")
        docs_train, docs_test, y_train, y_test = train_test_split(
            documents, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"Train: {len(docs_train)} | Test: {len(docs_test)}")

        # Extraction de features TF-IDF
        print("\nExtraction des features TF-IDF...")
        X_train = self.feature_extractor.fit_transform(docs_train)
        X_test = self.feature_extractor.transform(docs_test)
        print(f"Shape des features: {X_train.shape}")

        # Entraînement
        print(f"\nEntrainement du modele ({self.model_type})...")
        self.classifier.fit(X_train, y_train)

        # Évaluation
        print("\nEvaluation...")
        y_train_pred = self.classifier.predict(X_train)
        y_test_pred = self.classifier.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')

        print(f"Accuracy Train: {train_accuracy:.2%}")
        print(f"Accuracy Test: {test_accuracy:.2%}")
        print(f"F1-Score Test: {test_f1:.2%}")

        # Rapport détaillé
        print("\nClassification Report:")
        y_test_labels = self._decode_labels(y_test)
        y_pred_labels = self._decode_labels(y_test_pred)
        print(classification_report(y_test_labels, y_pred_labels))

        self._is_trained = True

        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'y_test': y_test_labels,
            'y_pred': y_pred_labels,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }

    def predict(self, text: str) -> str:
        """
        Prédit la catégorie d'un CV.

        Args:
            text: Texte du CV

        Returns:
            Catégorie prédite
        """
        if not self._is_trained:
            raise ValueError("Le modele doit etre entraine avant de predire")

        # Prétraitement
        tokens = self.preprocessor.preprocess(text)

        # Extraction de features
        features = self.feature_extractor.transform([tokens])

        # Prédiction
        prediction_idx = self.classifier.predict(features)[0]
        return self.idx_to_label[prediction_idx]

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Prédit les probabilités pour chaque catégorie.

        Args:
            text: Texte du CV

        Returns:
            Dictionnaire {catégorie: probabilité}
        """
        if not self._is_trained:
            raise ValueError("Le modele doit etre entraine avant de predire")

        if not hasattr(self.classifier, 'predict_proba'):
            raise ValueError(f"Le modele {self.model_type} ne supporte pas predict_proba")

        # Prétraitement
        tokens = self.preprocessor.preprocess(text)

        # Extraction de features
        features = self.feature_extractor.transform([tokens])

        # Probabilités
        probas = self.classifier.predict_proba(features)[0]

        return {
            self.idx_to_label[idx]: float(proba)
            for idx, proba in enumerate(probas)
        }

    def get_top_features_for_category(self, category: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Retourne les features les plus importantes pour une catégorie.

        Args:
            category: Nom de la catégorie
            n: Nombre de features à retourner

        Returns:
            Liste de tuples (feature, importance)
        """
        if category not in self.label_to_idx:
            raise ValueError(f"Categorie inconnue: {category}")

        feature_names = self.feature_extractor.get_feature_names()

        # Pour Naive Bayes, utiliser log probabilities
        if self.model_type == 'naive_bayes':
            cat_idx = self.label_to_idx[category]
            log_probs = self.classifier.feature_log_prob_[cat_idx]
            top_indices = np.argsort(log_probs)[::-1][:n]
            return [(feature_names[i], float(log_probs[i])) for i in top_indices]

        # Pour les autres modèles avec coef_
        elif hasattr(self.classifier, 'coef_'):
            cat_idx = self.label_to_idx[category]
            coefficients = self.classifier.coef_[cat_idx]
            top_indices = np.argsort(coefficients)[::-1][:n]
            return [(feature_names[i], float(coefficients[i])) for i in top_indices]

        return []

    def save(self, path: str) -> None:
        """Sauvegarde le modèle complet."""
        data = {
            'model_type': self.model_type,
            'classifier': self.classifier,
            'feature_extractor': self.feature_extractor,
            'preprocessor': self.preprocessor,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df
        }
        joblib.dump(data, path)
        print(f"Modele sauvegarde: {path}")

    @classmethod
    def load(cls, path: str) -> 'CVClassifier':
        """Charge un modèle sauvegardé."""
        data = joblib.load(path)

        instance = cls(
            model_type=data['model_type'],
            max_features=data['max_features'],
            ngram_range=data['ngram_range'],
            min_df=data['min_df']
        )

        instance.classifier = data['classifier']
        instance.feature_extractor = data['feature_extractor']
        instance.preprocessor = data['preprocessor']
        instance.label_to_idx = data['label_to_idx']
        instance.idx_to_label = data['idx_to_label']
        instance._is_trained = True

        print(f"Modele charge: {path}")
        return instance


def compare_models(
    csv_path: str,
    text_column: str = 'Resume',
    category_column: str = 'Category',
    max_features: int = 1500
) -> pd.DataFrame:
    """
    Compare les performances de différents modèles.

    Args:
        csv_path: Chemin vers le dataset
        text_column: Colonne du texte
        category_column: Colonne des catégories
        max_features: Nombre de features TF-IDF

    Returns:
        DataFrame avec les résultats comparatifs
    """
    models = ['naive_bayes', 'logistic_regression', 'svm', 'random_forest']
    results = []

    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Test du modele: {model_type.upper()}")
        print('='*60)

        classifier = CVClassifier(
            model_type=model_type,
            max_features=max_features
        )

        metrics = classifier.train(
            csv_path,
            text_column=text_column,
            category_column=category_column
        )

        results.append({
            'model': model_type,
            'train_accuracy': metrics['train_accuracy'],
            'test_accuracy': metrics['test_accuracy'],
            'test_f1': metrics['test_f1']
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('test_f1', ascending=False)

    print("\n" + "="*60)
    print("COMPARAISON DES MODELES")
    print("="*60)
    print(df_results.to_string(index=False))

    return df_results


# Test du module
if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/cv_dataset.csv"

    print("="*60)
    print("ENTRAINEMENT DU CLASSIFICATEUR DE CV")
    print("="*60 + "\n")

    # Entraîner avec Naive Bayes
    classifier = CVClassifier(
        model_type='naive_bayes',
        max_features=1500,
        ngram_range=(1, 2),
        language='both'
    )

    results = classifier.train(csv_path)

    # Test de prédiction
    print("\nTest de prediction:")
    sample_cv = """
    Data Scientist with 5 years of experience in machine learning,
    deep learning, and natural language processing. Proficient in
    Python, TensorFlow, PyTorch, and scikit-learn. Strong background
    in statistics and mathematics.
    """

    prediction = classifier.predict(sample_cv)
    print(f"CV de test -> Categorie predite: {prediction}")
