from pydantic import BaseModel


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
