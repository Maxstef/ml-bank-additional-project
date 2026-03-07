from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"