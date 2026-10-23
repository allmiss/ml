from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


def _wrap_single_client(value: Any) -> Any:
    return [value] if isinstance(value, dict) else value


# Тело /predict: один клиент или список клиентов. Один клиент сразу
# заворачивается в список, поэтому проверка идёт по одной схеме и ошибки
# указывают на конкретное поле, а не на обе ветки «объект | список».
FeatureVectorsChurn = Annotated[
    list[FeatureVectorChurn],
    BeforeValidator(_wrap_single_client, json_schema_input_type=FeatureVectorChurn | list[FeatureVectorChurn]),
]


class ChurnPrediction(BaseModel):
    churn: int
    probabilities: dict[int, float]


class PredictionResponseChurn(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "predictions": [
                {"churn": 1, "probabilities": {"0": 0.3278, "1": 0.6722}},
                {"churn": 0, "probabilities": {"0": 0.9807, "1": 0.0193}},
            ],
        }],
    })

    predictions: list[ChurnPrediction]


class ErrorResponse(BaseModel):
    """Общий формат ошибки: машинный код, сообщение для человека и подробности."""

    code: str
    message: str
    details: dict[str, Any] | None = None


class DatasetInfo(BaseModel):
    rows: int
    columns: int
    features: list[str]
    churn_distribution: dict[int, int]


class ClassDistribution(BaseModel):
    counts: dict[int, int]
    shares: dict[int, float]


class SplitInfo(BaseModel):
    train_size: int
    test_size: int
    churn_distribution: dict[str, ClassDistribution]


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
    # None у моделей, обученных до появления roc_auc
    roc_auc: float | None = None


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
