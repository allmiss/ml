import os
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model.joblib"
HISTORY_PATH = Path(__file__).parent.parent / "models" / "training_history.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
