from pathlib import Path

import joblib
from pydantic import ValidationError

from src.config import MODEL_PATH
from src.exceptions import InvalidModelFileError
from src.training import TrainedModel


def save_churn_model(model: TrainedModel, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_churn_model(path: Path = MODEL_PATH) -> TrainedModel:
    model = joblib.load(path)
    try:
        return TrainedModel(**{field: getattr(model, field) for field in TrainedModel.model_fields})
    except (AttributeError, ValidationError):
        raise InvalidModelFileError(
            f"Файл {path.name} в старом формате, переобучите модель через POST /model/train",
            details={"file": path.name},
        ) from None
