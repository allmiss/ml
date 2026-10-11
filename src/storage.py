from pathlib import Path

import joblib

from src.config import MODEL_PATH
from src.training import TrainedModel


def save_churn_model(model: TrainedModel, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_churn_model(path: Path = MODEL_PATH) -> TrainedModel:
    model = joblib.load(path)
    if not isinstance(model, TrainedModel):
        raise TypeError(f"Файл {path.name} в старом формате, переобучите модель через POST /model/train")
    return model
