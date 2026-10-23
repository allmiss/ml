from pathlib import Path

import joblib
from pydantic import ValidationError

from src.core.config import MODEL_PATH
from src.core.exceptions import InvalidModelFileError
from src.ml.training import TrainedModel


def save_churn_model(model: TrainedModel, path: Path = MODEL_PATH) -> None:
    # В файл — пайплайн sklearn и простые значения, без классов сервиса: pickle хранит
    # путь к классу, и после переноса модулей такой файл перестал бы загружаться
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model.pipeline, **model.model_dump(mode="json", exclude={"pipeline"})}, path)


def load_churn_model(path: Path = MODEL_PATH) -> TrainedModel:
    invalid = InvalidModelFileError(
        f"Файл {path.name} в неподдерживаемом формате, переобучите модель через POST /model/train",
        details={"file": path.name},
    )
    try:
        data = joblib.load(path)
    except OSError:
        # файла нет или его нельзя прочитать — это не формат, переобучение не поможет
        raise
    except Exception:
        # файл другой версии сервиса или повреждён: unpickle падает с разными исключениями
        raise invalid from None
    if not isinstance(data, dict):
        raise invalid
    try:
        return TrainedModel.model_validate(data)
    except ValidationError:
        raise invalid from None
