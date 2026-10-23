"""Ответы эндпоинтов /dataset/*."""

from pydantic import BaseModel


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
