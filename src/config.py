from pathlib import Path

DATASET_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42
