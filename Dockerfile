FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-root

RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
RUN python -m spacy download fr_core_news_sm || true
RUN python -m spacy download en_core_web_sm || true

COPY src/ ./src/
COPY frontend/ ./frontend/
COPY data/ ./data/

RUN mkdir -p ./data/models

EXPOSE 7860

CMD ["uvicorn", "cv_analyzer.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
