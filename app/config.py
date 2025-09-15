import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "news_classifier.pkl"

MODEL_DIR.mkdir(exist_ok=True)

API_TITLE = "News Classification API"
API_DESCRIPTION = "AI model for classifying news into politics, sports, and business categories"
API_VERSION = "1.0.0"

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]
