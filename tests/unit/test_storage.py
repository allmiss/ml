from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pytest

from src.core.exceptions import InvalidModelFileError
from src.ml.preprocessing import DataSplit
from src.ml.storage import load_churn_model, save_churn_model
from src.ml.training import TrainedModel, evaluate_model, resolve_config, train_churn_model
from src.schemas.model import TrainingConfigChurn


@pytest.fixture(scope="module")
def trained_model(churn_split: DataSplit) -> TrainedModel:
    X_train, X_test, y_train, y_test = churn_split
    config = resolve_config(TrainingConfigChurn())
    pipeline = train_churn_model(X_train, y_train, config)
    return TrainedModel(
        pipeline=pipeline,
        config=config,
        trained_at=datetime.now().astimezone(),
        metrics=evaluate_model(pipeline, X_test, y_test),
    )


def test_save_and_load(tmp_path: Path, trained_model: TrainedModel, churn_split: DataSplit) -> None:
    path = tmp_path / "model.joblib"
    save_churn_model(trained_model, path)

    loaded = load_churn_model(path)

    assert loaded.config == trained_model.config
    assert loaded.trained_at == trained_model.trained_at
    assert loaded.metrics == trained_model.metrics
    X_test = churn_split[1]
    assert (loaded.pipeline.predict_proba(X_test) == trained_model.pipeline.predict_proba(X_test)).all()


def test_file_does_not_reference_service_modules(tmp_path: Path, trained_model: TrainedModel) -> None:
    # pickle хранит пути к классам: если в файле есть src.*, перенос модулей сломает загрузку
    path = tmp_path / "model.joblib"
    save_churn_model(trained_model, path)
    assert b"src." not in path.read_bytes()


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_churn_model(tmp_path / "nope.joblib")


def test_unreadable_file_is_not_called_unsupported(tmp_path: Path) -> None:
    # на месте файла папка: это ошибка доступа, а не формат — «переобучите» тут не поможет
    path = tmp_path / "model.joblib"
    path.mkdir()
    with pytest.raises(OSError):
        load_churn_model(path)


@pytest.mark.parametrize(
    "content",
    [b"not a pickle", ["something", "else"], {"pipeline": None, "config": {}}],
    ids=["broken_file", "not_a_dict", "dict_without_fields"],
)
def test_unsupported_file(tmp_path: Path, content: Any) -> None:
    path = tmp_path / "model.joblib"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        joblib.dump(content, path)
    with pytest.raises(InvalidModelFileError, match="переобучите"):
        load_churn_model(path)
