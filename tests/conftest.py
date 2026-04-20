"""Fixtures pytest pour CV Analyzer."""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from cv_analyzer.api.main import app
from cv_analyzer.api.dependencies import ClassifierService, get_settings


# Sample CV texts for testing
SAMPLE_CV_SHORT = "This is a short CV with not enough content."

SAMPLE_CV_DATA_SCIENTIST = """
John Smith - Senior Data Scientist

PROFESSIONAL SUMMARY
Experienced Data Scientist with 5+ years of expertise in machine learning,
deep learning, and natural language processing. Proven track record of
delivering data-driven solutions that drive business growth.

EXPERIENCE

Senior Data Scientist | Tech Corp | 2021 - Present
- Developed machine learning models for customer churn prediction
- Implemented NLP pipelines for sentiment analysis
- Led a team of 3 junior data scientists
- Achieved 25% improvement in model accuracy

Data Scientist | Analytics Inc | 2019 - 2021
- Built recommendation systems using collaborative filtering
- Created automated reporting dashboards with Python and SQL
- Collaborated with product teams to define KPIs

EDUCATION
Master of Science in Computer Science - Stanford University, 2019
Bachelor of Science in Mathematics - MIT, 2017

SKILLS
Python, R, SQL, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy
Machine Learning, Deep Learning, NLP, Computer Vision
Docker, Kubernetes, AWS, GCP, Apache Spark
"""

SAMPLE_CV_SOFTWARE_ENGINEER = """
Jane Doe - Software Engineer

SUMMARY
Full-stack software engineer with strong experience in web development
and cloud technologies. Passionate about building scalable applications.

WORK EXPERIENCE

Software Engineer | Google | 2020 - Present
- Developed microservices using Java and Spring Boot
- Built responsive web interfaces with React and TypeScript
- Implemented CI/CD pipelines with Jenkins and Docker
- Mentored junior developers

Junior Developer | Startup XYZ | 2018 - 2020
- Created REST APIs with Node.js and Express
- Managed PostgreSQL databases
- Participated in agile development processes

EDUCATION
Bachelor in Computer Engineering - UC Berkeley, 2018

TECHNICAL SKILLS
Languages: Java, Python, JavaScript, TypeScript
Frameworks: Spring Boot, React, Node.js, Django
Databases: PostgreSQL, MongoDB, Redis
Cloud: AWS, GCP, Docker, Kubernetes
"""


@pytest.fixture
def sample_cv_short():
    """CV trop court pour analyse."""
    return SAMPLE_CV_SHORT


@pytest.fixture
def sample_cv_data_scientist():
    """CV de Data Scientist."""
    return SAMPLE_CV_DATA_SCIENTIST


@pytest.fixture
def sample_cv_software_engineer():
    """CV de Software Engineer."""
    return SAMPLE_CV_SOFTWARE_ENGINEER


@pytest.fixture
def test_client():
    """Client de test FastAPI."""
    return TestClient(app)


@pytest.fixture
def model_path():
    """Chemin vers le modele de classification."""
    settings = get_settings()
    return settings.model_path


@pytest.fixture
def classifier_loaded(model_path):
    """Assure que le classificateur est charge."""
    if not ClassifierService.is_loaded():
        ClassifierService.load(model_path)
    return ClassifierService.is_loaded()
