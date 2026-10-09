"""NLP components of the University Administrative Copilot (intent, NER, retrieval, pipeline)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"
DATA = PROJECT_ROOT / "data"
