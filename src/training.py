from datetime import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.models import TrainMetrics
from src.preprocessing import create_preprocessor


class TrainedModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline: Pipeline
    trained_at: datetime
    metrics: TrainMetrics


def create_model() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def create_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", create_model()),
    ])


def train_churn_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> TrainMetrics:
    y_pred = pipeline.predict(X_test)
    return TrainMetrics(
        accuracy=round(accuracy_score(y_test, y_pred), 3),
        f1=round(f1_score(y_test, y_pred), 3),
    )
