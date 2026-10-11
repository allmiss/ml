from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FeatureVectorChurn(BaseModel):
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
    detail: str


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


class TrainMetrics(BaseModel):
    accuracy: float
    f1: float


class ModelStatus(BaseModel):
    trained: bool
    trained_at: datetime | None = None
    metrics: TrainMetrics | None = None
