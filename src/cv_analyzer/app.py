"""
Interface Gradio pour l'analyse de CV
Supporte le chargement de fichiers (PDF, Word, TXT) et le texte brut
"""

import gradio as gr
from typing import Optional, Tuple
import os

from .file_loader import FileLoader
from .classifier import CVClassifier
from .summarizer import CVSummarizer


class CVAnalyzerApp:
    """
    Application Gradio pour l'analyse de CV.

    Fonctionnalités:
    - Chargement de fichiers (PDF, Word, TXT)
    - Saisie de texte brut
    - Classification du CV
    - Génération de résumé extractif
    """

    def __init__(
        self,
        classifier: Optional[CVClassifier] = None,
        summarizer_method: str = 'tfidf'
    ):
        """
        Initialise l'application.

        Args:
            classifier: Classificateur pré-entraîné (ou None)
            summarizer_method: Méthode de résumé ('tfidf', 'textrank', 'frequency')
        """
        self.classifier = classifier
        self.file_loader = FileLoader()
        self.summarizer = CVSummarizer(method=summarizer_method)
        self._classifier_loaded = classifier is not None

    def load_classifier(self, model_path: str) -> bool:
        """Charge un classificateur depuis un fichier."""
        try:
            self.classifier = CVClassifier.load(model_path)
            self._classifier_loaded = True
            return True
        except Exception as e:
            print(f"Erreur chargement modele: {e}")
            return False

    def extract_text_from_file(self, file) -> Tuple[str, str]:
        """
        Extrait le texte d'un fichier uploadé.

        Args:
            file: Fichier uploadé via Gradio

        Returns:
            Tuple (texte extrait, message de statut)
        """
        if file is None:
            return "", "Aucun fichier selectionne"

        try:
            # Gradio fournit le chemin du fichier temporaire
            file_path = file.name if hasattr(file, 'name') else str(file)

            # Vérifier si le format est supporté
            if not self.file_loader.is_supported(file_path):
                ext = os.path.splitext(file_path)[1]
                return "", f"Format non supporte: {ext}. Utilisez PDF, DOCX ou TXT."

            # Extraire le texte
            text = self.file_loader.load(file_path)

            if not text or len(text.strip()) < 50:
                return "", "Le fichier semble vide ou trop court"

            return text, f"Fichier charge ({len(text)} caracteres)"

        except Exception as e:
            return "", f"Erreur: {str(e)}"

    def analyze_cv(
        self,
        text: str,
        num_sentences: int = 5
    ) -> str:
        """
        Analyse un CV (classification + résumé).

        Args:
            text: Texte du CV
            num_sentences: Nombre de phrases pour le résumé

        Returns:
            Résultat formaté en Markdown
        """
        if not text or len(text.strip()) < 50:
            return "Veuillez fournir un CV (minimum 50 caracteres)"

        results = []

        # Classification
        if self._classifier_loaded and self.classifier:
            try:
                category = self.classifier.predict(text)
                results.append(f"## Categorie Predite\n**{category}**")
            except Exception as e:
                results.append(f"## Classification\nErreur: {str(e)}")
        else:
            results.append("## Classification\n*Modele non charge - Entrainez d'abord le classificateur*")

        # Résumé
        try:
            summary = self.summarizer.summarize(text, num_sentences=num_sentences)
            results.append(f"## Resume du CV\n{summary}")
        except Exception as e:
            results.append(f"## Resume\nErreur: {str(e)}")

        # Statistiques
        word_count = len(text.split())
        char_count = len(text)
        results.append(f"\n---\n*{word_count} mots | {char_count} caracteres*")

        return "\n\n".join(results)

    def process_input(
        self,
        file,
        text_input: str,
        num_sentences: int
    ) -> Tuple[str, str]:
        """
        Traite l'entrée (fichier ou texte).

        Args:
            file: Fichier uploadé (peut être None)
            text_input: Texte saisi manuellement
            num_sentences: Nombre de phrases pour le résumé

        Returns:
            Tuple (texte extrait, résultat de l'analyse)
        """
        # Priorité au fichier si fourni
        if file is not None:
            text, status = self.extract_text_from_file(file)
            if not text:
                return "", status
        else:
            text = text_input
            status = "Texte saisi manuellement"

        if not text.strip():
            return "", "Veuillez charger un fichier ou saisir du texte"

        # Analyser
        result = self.analyze_cv(text, num_sentences)

        return text, result

    def create_interface(self) -> gr.Blocks:
        """
        Crée l'interface Gradio.

        Returns:
            Interface Gradio
        """
        with gr.Blocks(title="CV Analyzer") as interface:
            gr.Markdown("""
            # CV Analyzer
            Analyseur de CV avec classification et resume automatique

            **Fonctionnalites:**
            - Chargement de fichiers PDF, Word (.docx), TXT
            - Classification par categorie professionnelle
            - Resume extractif (TF-IDF, TextRank, Frequence)
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    # Entrées
                    file_input = gr.File(
                        label="Charger un CV (PDF, DOCX, TXT)",
                        file_types=[".pdf", ".docx", ".txt"]
                    )

                    text_input = gr.Textbox(
                        label="Ou collez le texte du CV",
                        lines=10,
                        placeholder="Collez le contenu du CV ici..."
                    )

                    num_sentences = gr.Slider(
                        minimum=3,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Nombre de phrases dans le resume"
                    )

                    analyze_btn = gr.Button("Analyser le CV", variant="primary")

                with gr.Column(scale=1):
                    # Sorties
                    extracted_text = gr.Textbox(
                        label="Texte extrait",
                        lines=8,
                        interactive=False
                    )

                    result_output = gr.Markdown(
                        label="Resultat de l'analyse"
                    )

            # Exemple de CV
            gr.Markdown("""
            ---
            **Exemple de CV a copier-coller:**
            ```
            John Smith - Software Engineer

            EXPERIENCE
            Senior Software Engineer at Tech Corp (2020-2024)
            - Developed microservices architecture using Python and Docker
            - Led a team of 5 developers on cloud migration project

            EDUCATION
            Master in Computer Science - MIT (2018)

            SKILLS
            Python, Java, JavaScript, AWS, Docker, Kubernetes, SQL
            ```
            """)

            # Événements
            analyze_btn.click(
                fn=self.process_input,
                inputs=[file_input, text_input, num_sentences],
                outputs=[extracted_text, result_output]
            )

            # Mise à jour automatique lors du chargement de fichier
            file_input.change(
                fn=self.extract_text_from_file,
                inputs=[file_input],
                outputs=[extracted_text, result_output]
            )

        return interface


def create_app(
    model_path: Optional[str] = None,
    summarizer_method: str = 'tfidf'
) -> gr.Blocks:
    """
    Crée l'application CV Analyzer.

    Args:
        model_path: Chemin vers le modèle de classification (optionnel)
        summarizer_method: Méthode de résumé

    Returns:
        Interface Gradio
    """
    app = CVAnalyzerApp(summarizer_method=summarizer_method)

    if model_path and os.path.exists(model_path):
        app.load_classifier(model_path)
        print(f"Modele charge: {model_path}")
    else:
        print("Aucun modele de classification charge")
        print("Le resume fonctionnera, mais pas la classification")

    return app.create_interface()


# Point d'entrée
if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("CV ANALYZER - Interface Web")
    print("=" * 60)

    interface = create_app(model_path=model_path)
    interface.launch(share=False, theme=gr.themes.Soft())
