"""Конфигурация обучения, метрики, история и ответы эндпоинтов /model/*."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FeatureInfo(BaseModel):
    name: str
    type: str
    kind: Literal["numeric", "categorical"]
    allowed_values: list[str] | None = None


class ModelSchema(BaseModel):
    features: list[FeatureInfo]
    target: str


class TrainingConfigChurn(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {"model_type": "logreg", "hyperparameters": {"C": 1.0, "class_weight": "balanced"}},
            {"model_type": "random_forest", "hyperparameters": {"n_estimators": 200, "max_depth": 8}},
        ],
    })

    model_type: str = "logreg"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class TrainMetrics(BaseModel):
    accuracy: float
    f1: float
    roc_auc: float


class TrainingRecord(BaseModel):
    """Одна запись истории обучений: какая модель, на каких данных и с каким качеством."""

    id: int
    trained_at: datetime
    model_type: str
    hyperparameters: dict[str, Any]
    train_size: int
    test_size: int
    metrics: TrainMetrics


class ModelMetrics(BaseModel):
    """Метрики последнего обучения и, по запросу, несколько последних записей истории."""

    latest: TrainingRecord | None
    history: list[TrainingRecord]


MetricName = Literal["accuracy", "f1", "roc_auc"]


class ConfigComparison(BaseModel):
    """Одна настройка модели (тип + гиперпараметры) и её метрики по последнему обучению с ней."""

    model_type: str
    hyperparameters: dict[str, Any]
    metrics: TrainMetrics
    runs: int
    last_record_id: int
    last_trained_at: datetime


class ModelComparison(BaseModel):
    """Настройки модели из истории обучений, от лучшей к худшей по выбранной метрике."""

    metric: MetricName
    best: ConfigComparison | None
    configs: list[ConfigComparison]


class ModelStatus(BaseModel):
    trained: bool
    model_type: str | None = None
    hyperparameters: dict[str, Any] | None = None
    trained_at: datetime | None = None
    metrics: TrainMetrics | None = None
