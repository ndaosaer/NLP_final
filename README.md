# CV Analyzer

Application d'analyse et de classification de CV utilisant des techniques NLP classiques.

## Fonctionnalites

- **Classification de CV** : Categorise automatiquement les CV (Data Science, Software Engineering, etc.)
- **Resume extractif** : Genere un resume des points cles du CV (TF-IDF, TextRank, Frequence)
- **Extraction de texte** : Supporte PDF, DOCX et TXT
- **API REST** : API FastAPI complete avec documentation OpenAPI
- **Interface Web** : Frontend moderne en HTML/CSS/JS

## Architecture

```
projet-nlp/
├── src/cv_analyzer/          # Module principal
│   ├── api/                  # API FastAPI
│   │   ├── main.py          # Point d'entree
│   │   ├── routes/          # Endpoints
│   │   ├── schemas.py       # Modeles Pydantic
│   │   └── dependencies.py  # Services
│   ├── classifier.py        # Classification NLP
│   ├── summarizer.py        # Resume extractif
│   ├── preprocessing.py     # Preprocessing texte
│   ├── features.py          # TF-IDF, BoW
│   └── file_loader.py       # Chargement fichiers
├── frontend/                 # Interface web
├── tests/                    # Tests (unit, integration, e2e)
├── infra/                    # Docker, ECS
└── .github/workflows/        # CI/CD
```

## Installation

### Prerequis

- Python 3.12
- Poetry 2.0+

### Installation locale

```bash
# Cloner le repo
git clone <repo-url>
cd projet-nlp

# Installer les dependances
poetry install

# Telecharger les donnees NLTK
poetry run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Utilisation

### Lancer l'API

```bash
# Mode developpement
poetry run uvicorn cv_analyzer.api.main:app --reload

# L'API est disponible sur http://localhost:8000
# Documentation: http://localhost:8000/docs
```

### Lancer avec Docker

```bash
cd infra
docker-compose up --build

# API: http://localhost:8000
# Frontend: http://localhost:80
```

### Endpoints API

| Methode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/info` | Info API |
| POST | `/api/v1/classify` | Classifier un CV |
| POST | `/api/v1/classify/proba` | Classification avec probabilites |
| GET | `/api/v1/categories` | Liste des categories |
| POST | `/api/v1/summarize` | Resumer un CV |
| GET | `/api/v1/summarize/methods` | Methodes disponibles |
| POST | `/api/v1/analyze` | Analyse complete |
| POST | `/api/v1/extract-text` | Extraire texte d'un fichier |

### Exemple d'utilisation

```python
import requests

# Classifier un CV
response = requests.post(
    "http://localhost:8000/api/v1/classify",
    json={"text": "Data Scientist with 5 years experience..."}
)
print(response.json())
# {"category": "Data Science", "confidence": 0.85}

# Analyse complete
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={
        "text": "CV content...",
        "classify": True,
        "summarize": True,
        "method": "tfidf",
        "num_sentences": 5
    }
)
```

## Tests

```bash
# Lancer tous les tests
poetry run pytest

# Avec couverture
poetry run pytest --cov=cv_analyzer --cov-report=html

# Tests specifiques
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest tests/e2e/
```

## CI/CD

### GitHub Actions

- **CI** (`ci.yml`): Lint (Ruff), Tests, Build Docker
- **CD** (`cd.yml`): Deploy vers AWS ECS Fargate

### Secrets requis

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
API_BASE_URL
S3_BUCKET_NAME
CLOUDFRONT_DISTRIBUTION_ID
```

## Deploiement AWS

### Architecture

```
CloudFront (Frontend S3)
         │
         ▼
Application Load Balancer
         │
         ▼
    ECS Fargate
    (API Containers)
```

### Deploiement manuel

```bash
# Build et push vers ECR
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.eu-west-1.amazonaws.com
docker build -t cv-analyzer -f infra/Dockerfile .
docker tag cv-analyzer:latest <account>.dkr.ecr.eu-west-1.amazonaws.com/cv-analyzer:latest
docker push <account>.dkr.ecr.eu-west-1.amazonaws.com/cv-analyzer:latest

# Deployer le frontend
aws s3 sync frontend/ s3://<bucket-name>/
```

## Technologies

- **Backend**: FastAPI, Uvicorn
- **NLP**: NLTK, scikit-learn
- **ML**: PyTorch, Transformers
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Infrastructure**: Docker, AWS ECS Fargate, S3, CloudFront
- **CI/CD**: GitHub Actions

## License

MIT
