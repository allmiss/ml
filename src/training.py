from datetime import datetime
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.exceptions import InvalidTrainingConfigError
from src.models import TrainingConfigChurn, TrainMetrics
from src.preprocessing import create_preprocessor

MODEL_CLASSES: dict[str, type[ClassifierMixin]] = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
}

# Поддерживаемые типы моделей как перечисление — для параметров запросов,
# где неизвестный тип должен сразу давать 422 со списком допустимых значений
ModelType = StrEnum("ModelType", {name: name for name in MODEL_CLASSES})

DEFAULT_HYPERPARAMETERS: dict[str, dict] = {
    "logreg": {"max_iter": 1000},
    "random_forest": {"random_state": RANDOM_STATE},
}


class TrainedModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline: Pipeline
    config: TrainingConfigChurn
    trained_at: datetime
    metrics: TrainMetrics


def resolve_config(config: TrainingConfigChurn) -> TrainingConfigChurn:
    if config.model_type not in MODEL_CLASSES:
        raise InvalidTrainingConfigError(
            f"Неизвестный тип модели '{config.model_type}'",
            details={"model_type": config.model_type, "supported": list(MODEL_CLASSES)},
        )
    return TrainingConfigChurn(
        model_type=config.model_type,
        hyperparameters={**DEFAULT_HYPERPARAMETERS[config.model_type], **config.hyperparameters},
    )


def create_model(config: TrainingConfigChurn) -> ClassifierMixin:
    config = resolve_config(config)
    try:
        return MODEL_CLASSES[config.model_type](**config.hyperparameters)
    except TypeError as e:
        raise InvalidTrainingConfigError(
            f"Неверные гиперпараметры для {config.model_type}",
            details={"model_type": config.model_type, "reason": str(e)},
        ) from None


def create_pipeline(config: TrainingConfigChurn) -> Pipeline:
    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", create_model(config)),
    ])


def train_churn_model(X_train: pd.DataFrame, y_train: pd.Series, config: TrainingConfigChurn) -> Pipeline:
    pipeline = create_pipeline(config)
    try:
        pipeline.fit(X_train, y_train)
    except ValueError as e:
        raise InvalidTrainingConfigError(
            f"Не удалось обучить {config.model_type}",
            details={"model_type": config.model_type, "reason": str(e)},
        ) from None
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> TrainMetrics:
    y_pred = pipeline.predict(X_test)
    churn_proba = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
    return TrainMetrics(
        accuracy=round(accuracy_score(y_test, y_pred), 3),
        f1=round(f1_score(y_test, y_pred), 3),
        roc_auc=round(roc_auc_score(y_test, churn_proba), 3),
    )
