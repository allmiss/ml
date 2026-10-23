from pathlib import Path

import joblib
from pydantic import ValidationError

from src.config import MODEL_PATH
from src.exceptions import InvalidModelFileError
from src.models import TrainMetrics
from src.training import TrainedModel


def save_churn_model(model: TrainedModel, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_churn_model(path: Path = MODEL_PATH) -> TrainedModel:
    model = joblib.load(path)
    try:
        fields = {field: getattr(model, field) for field in TrainedModel.model_fields}
        # Метрики пересобираем, чтобы у моделей, сохранённых до появления
        # новых метрик (roc_auc), эти поля получили значения по умолчанию
        fields["metrics"] = TrainMetrics(**vars(fields["metrics"]))
        return TrainedModel(**fields)
    except (AttributeError, ValidationError):
        raise InvalidModelFileError(
            f"Файл {path.name} в старом формате, переобучите модель через POST /model/train",
            details={"file": path.name},
        ) from None
