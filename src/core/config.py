import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = SRC_DIR.parent

DATASET_PATH = SRC_DIR / "data" / "churn_dataset.csv"
MODEL_PATH = PROJECT_DIR / "models" / "churn_model.joblib"
HISTORY_PATH = PROJECT_DIR / "models" / "training_history.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
