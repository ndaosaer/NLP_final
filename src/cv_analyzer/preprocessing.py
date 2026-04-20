"""
Module de prétraitement NLP pour les CV
Techniques classiques : tokenisation, lemmatisation, stopwords, normalisation
"""

import re
import string
import unicodedata
from typing import List, Optional


class TextPreprocessor:
    """
    Pipeline de prétraitement NLP classique pour les CV.

    Étapes du pipeline:
    1. Normalisation (lowercase, accents)
    2. Nettoyage (URLs, emails, caractères spéciaux)
    3. Tokenisation
    4. Suppression des stopwords
    5. Lemmatisation
    6. Filtrage (longueur minimale, chiffres)
    """

    # Stopwords français courants
    FRENCH_STOPWORDS = {
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'en', 'au', 'aux',
        'ce', 'ces', 'cette', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
        'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
        'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
        'me', 'te', 'se', 'lui', 'y', 'qui', 'que', 'quoi', 'dont', 'où',
        'ne', 'pas', 'plus', 'jamais', 'rien', 'personne',
        'être', 'avoir', 'faire', 'pouvoir', 'vouloir', 'devoir', 'aller',
        'est', 'sont', 'était', 'été', 'ai', 'as', 'a', 'avons', 'avez', 'ont',
        'suis', 'es', 'sommes', 'êtes',
        'pour', 'par', 'sur', 'sous', 'dans', 'avec', 'sans', 'entre', 'vers', 'chez',
        'mais', 'ou', 'donc', 'car', 'ni', 'si', 'comme', 'quand', 'lorsque',
        'très', 'bien', 'aussi', 'ainsi', 'alors', 'encore', 'toujours', 'déjà',
        'tout', 'tous', 'toute', 'toutes', 'autre', 'autres', 'même', 'mêmes',
        'peu', 'beaucoup', 'trop', 'assez', 'moins', 'plus',
        'ici', 'là', 'ceci', 'cela', 'celui', 'celle', 'ceux', 'celles',
        'chaque', 'quelque', 'quelques', 'certain', 'certains', 'certaine', 'certaines',
        'tel', 'telle', 'tels', 'telles', 'quel', 'quelle', 'quels', 'quelles',
    }

    # Stopwords anglais courants
    ENGLISH_STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves',
        'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
        'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        'am', 'being', 'having', 'doing',
        'if', 'then', 'else', 'when', 'where', 'why', 'how', 'all', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
        'here', 'there', 'about', 'above', 'after', 'again', 'against', 'below',
        'between', 'both', 'during', 'into', 'through', 'under', 'until', 'while',
    }

    def __init__(
        self,
        language: str = 'both',
        remove_accents: bool = True,
        min_token_length: int = 2,
        remove_numbers: bool = False,
        use_lemmatization: bool = True,
        custom_stopwords: Optional[set] = None
    ):
        """
        Initialise le préprocesseur.

        Args:
            language: 'french', 'english' ou 'both' pour les stopwords
            remove_accents: Supprimer les accents
            min_token_length: Longueur minimale des tokens
            remove_numbers: Supprimer les tokens numériques
            use_lemmatization: Utiliser la lemmatisation (nécessite NLTK/SpaCy)
            custom_stopwords: Stopwords personnalisés à ajouter
        """
        self.language = language
        self.remove_accents = remove_accents
        self.min_token_length = min_token_length
        self.remove_numbers = remove_numbers
        self.use_lemmatization = use_lemmatization

        # Construction de la liste des stopwords
        self.stopwords = self._build_stopwords(custom_stopwords)

        # Initialisation des outils NLP
        self._nltk_lemmatizer = None
        self._spacy_nlp = None
        self._init_nlp_tools()

    def _build_stopwords(self, custom_stopwords: Optional[set]) -> set:
        """Construit la liste des stopwords selon la langue."""
        stopwords = set()

        if self.language in ('french', 'both'):
            stopwords.update(self.FRENCH_STOPWORDS)
        if self.language in ('english', 'both'):
            stopwords.update(self.ENGLISH_STOPWORDS)
        if custom_stopwords:
            stopwords.update(custom_stopwords)

        return stopwords

    def _init_nlp_tools(self) -> None:
        """Initialise les outils NLP (NLTK, SpaCy)."""
        if not self.use_lemmatization:
            return

        # Essayer NLTK d'abord
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer

            # Télécharger les ressources nécessaires
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                print("Téléchargement de WordNet...")
                nltk.download('wordnet', quiet=True)

            try:
                nltk.data.find('corpora/omw-1.4')
            except LookupError:
                nltk.download('omw-1.4', quiet=True)

            self._nltk_lemmatizer = WordNetLemmatizer()
            print("NLTK WordNet lemmatizer initialisé.")
            return

        except ImportError:
            pass

        # Essayer SpaCy ensuite
        try:
            import spacy

            # Essayer le modèle français
            try:
                self._spacy_nlp = spacy.load('fr_core_news_sm')
                print("SpaCy (français) initialisé.")
            except OSError:
                # Essayer le modèle anglais
                try:
                    self._spacy_nlp = spacy.load('en_core_web_sm')
                    print("SpaCy (anglais) initialisé.")
                except OSError:
                    print("Aucun modèle SpaCy trouvé. Lemmatisation désactivée.")
                    print("Installer avec: python -m spacy download fr_core_news_sm")

        except ImportError:
            print("Ni NLTK ni SpaCy installé. Lemmatisation désactivée.")

    def normalize(self, text: str) -> str:
        """
        Normalise le texte (lowercase, accents, unicode).

        Args:
            text: Texte brut

        Returns:
            Texte normalisé
        """
        # Conversion en minuscules
        text = text.lower()

        # Suppression des accents si demandé
        if self.remove_accents:
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(
                char for char in text
                if not unicodedata.combining(char)
            )

        return text

    def clean(self, text: str) -> str:
        """
        Nettoie le texte (URLs, emails, caractères spéciaux).

        Args:
            text: Texte à nettoyer

        Returns:
            Texte nettoyé
        """
        # Suppression des URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

        # Suppression des emails
        text = re.sub(r'\S+@\S+\.\S+', ' ', text)

        # Suppression des numéros de téléphone
        text = re.sub(r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}', ' ', text)

        # Suppression des caractères spéciaux (garder lettres, chiffres, espaces)
        text = re.sub(r'[^\w\s]', ' ', text)

        # Suppression des underscores
        text = re.sub(r'_', ' ', text)

        # Normalisation des espaces multiples
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenise le texte en mots.

        Args:
            text: Texte à tokeniser

        Returns:
            Liste de tokens
        """
        # Tokenisation simple par espaces
        tokens = text.split()

        # Filtrage par longueur minimale
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        # Filtrage des nombres si demandé
        if self.remove_numbers:
            tokens = [t for t in tokens if not t.isdigit()]

        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Supprime les stopwords.

        Args:
            tokens: Liste de tokens

        Returns:
            Tokens sans stopwords
        """
        return [t for t in tokens if t not in self.stopwords]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatise les tokens (réduit au lemme).

        Args:
            tokens: Liste de tokens

        Returns:
            Tokens lemmatisés
        """
        if not self.use_lemmatization:
            return tokens

        # Utiliser NLTK si disponible
        if self._nltk_lemmatizer:
            return [self._nltk_lemmatizer.lemmatize(token) for token in tokens]

        # Utiliser SpaCy si disponible
        if self._spacy_nlp:
            text = ' '.join(tokens)
            doc = self._spacy_nlp(text)
            return [token.lemma_ for token in doc if token.text in tokens]

        # Pas de lemmatisation disponible
        return tokens

    def preprocess(self, text: str) -> List[str]:
        """
        Applique le pipeline complet de prétraitement.

        Args:
            text: Texte brut du CV

        Returns:
            Liste de tokens prétraités
        """
        # 1. Normalisation
        text = self.normalize(text)

        # 2. Nettoyage
        text = self.clean(text)

        # 3. Tokenisation
        tokens = self.tokenize(text)

        # 4. Suppression des stopwords
        tokens = self.remove_stopwords(tokens)

        # 5. Lemmatisation
        tokens = self.lemmatize(tokens)

        return tokens

    def preprocess_to_text(self, text: str) -> str:
        """
        Prétraite et retourne le texte reconstruit.

        Args:
            text: Texte brut

        Returns:
            Texte prétraité (tokens joints par espaces)
        """
        tokens = self.preprocess(text)
        return ' '.join(tokens)

    def get_stats(self, text: str) -> dict:
        """
        Calcule des statistiques sur le texte.

        Args:
            text: Texte brut

        Returns:
            Dictionnaire de statistiques
        """
        tokens_raw = self.tokenize(self.clean(self.normalize(text)))
        tokens_processed = self.preprocess(text)

        return {
            'caracteres_original': len(text),
            'mots_avant_traitement': len(tokens_raw),
            'mots_apres_traitement': len(tokens_processed),
            'mots_supprimes': len(tokens_raw) - len(tokens_processed),
            'taux_reduction': round(
                (1 - len(tokens_processed) / max(len(tokens_raw), 1)) * 100, 2
            ),
            'vocabulaire_unique': len(set(tokens_processed))
        }


def preprocess_cv(text: str, **kwargs) -> List[str]:
    """
    Fonction utilitaire pour prétraiter un CV.

    Args:
        text: Texte du CV
        **kwargs: Arguments pour TextPreprocessor

    Returns:
        Liste de tokens prétraités
    """
    preprocessor = TextPreprocessor(**kwargs)
    return preprocessor.preprocess(text)


# Pour tester le module
if __name__ == "__main__":
    # Exemple de CV
    sample_cv = """
    Jean DUPONT
    Email: jean.dupont@email.com
    Tél: +33 6 12 34 56 78
    LinkedIn: https://linkedin.com/in/jeandupont

    EXPÉRIENCE PROFESSIONNELLE

    Data Scientist Senior - Entreprise ABC (2020-2023)
    - Développement de modèles de machine learning pour la prédiction des ventes
    - Mise en place de pipelines de données avec Python et Apache Spark
    - Collaboration avec les équipes métier pour définir les KPIs

    FORMATION

    Master en Intelligence Artificielle - Université XYZ (2018-2020)
    Licence en Mathématiques - Université ABC (2015-2018)

    COMPÉTENCES

    Python, R, SQL, TensorFlow, PyTorch, Scikit-learn
    Machine Learning, Deep Learning, NLP, Computer Vision
    """

    print("=== Test du préprocesseur NLP ===\n")

    preprocessor = TextPreprocessor(
        language='both',
        remove_accents=True,
        min_token_length=2,
        use_lemmatization=True
    )

    print("Texte original:")
    print(sample_cv[:500] + "...")
    print()

    tokens = preprocessor.preprocess(sample_cv)
    print("Tokens prétraités:")
    print(tokens[:50])
    print()

    stats = preprocessor.get_stats(sample_cv)
    print("Statistiques:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
