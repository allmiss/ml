import json
from pathlib import Path
from typing import Any

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import src.main
from src.models import TrainingConfigChurn
from src.preprocessing import DataSplit, create_preprocessor
from src.routes.prediction import LOYAL_CLIENT, RISKY_CLIENT
from src.training import create_pipeline
from tests.api.helpers import assert_error

JSON_HEADERS = {"Content-Type": "application/json"}


def test_single_client_same_as_in_batch(trained_client: TestClient) -> None:
    single = trained_client.post("/predict", json=LOYAL_CLIENT).json()["predictions"]
    batch = trained_client.post("/predict", json=[RISKY_CLIENT, LOYAL_CLIENT]).json()["predictions"]
    assert single == [batch[1]]


# --- нет модели ---

def test_without_trained_model(client: TestClient) -> None:
    body = assert_error(client.post("/predict", json=RISKY_CLIENT), 503, "model_not_trained")
    assert "/model/train" in body["message"]


def test_old_model_file_service_starts_without_model(model_dir: Path, churn_split: DataSplit) -> None:
    # в файле лежит голый пайплайн, как сохраняли до появления TrainedModel
    X_train, _, y_train, _ = churn_split
    joblib.dump(create_pipeline(TrainingConfigChurn()).fit(X_train, y_train), model_dir / "churn_model.joblib")

    with TestClient(src.main.app) as client:
        assert client.get("/model/status").json()["trained"] is False
        assert_error(client.post("/predict", json=RISKY_CLIENT), 503, "model_not_trained")


# --- неверные данные клиента ---

@pytest.mark.parametrize(
    ("body", "field", "error_type"),
    [
        ({k: v for k, v in RISKY_CLIENT.items() if k != "region"}, "0.region", "missing"),
        ({**RISKY_CLIENT, "extra": 1}, "0.extra", "extra_forbidden"),
        ({**RISKY_CLIENT, "monthly_fee": "abc"}, "0.monthly_fee", "float_parsing"),
        ({**RISKY_CLIENT, "support_requests": 1.5}, "0.support_requests", "int_from_float"),
        ({**RISKY_CLIENT, "region": None}, "0.region", "string_type"),
        ([RISKY_CLIENT, {**RISKY_CLIENT, "usage_hours": "x"}], "1.usage_hours", "float_parsing"),
        (42, "body", "list_type"),
    ],
    ids=["missing_feature", "extra_feature", "not_a_number", "float_for_int", "null", "second_client", "not_object"],
)
def test_invalid_client(trained_client: TestClient, body: Any, field: str, error_type: str) -> None:
    response = trained_client.post("/predict", json=body)
    errors = assert_error(response, 422, "validation_error")["details"]["errors"]
    assert len(errors) == 1
    assert (errors[0]["field"], errors[0]["type"]) == (field, error_type)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e400"])
def test_not_finite_number(trained_client: TestClient, value: str) -> None:
    # json.dumps такие значения не пишет, поэтому подставляем их в текст запроса
    body = json.dumps({**RISKY_CLIENT, "monthly_fee": "VALUE"}).replace('"VALUE"', value)
    response = trained_client.post("/predict", content=body, headers=JSON_HEADERS)
    error = assert_error(response, 422, "validation_error")["details"]["errors"][0]
    assert error["type"] == "finite_number"


def test_invalid_json(trained_client: TestClient) -> None:
    response = trained_client.post("/predict", content="{bad json", headers=JSON_HEADERS)
    error = assert_error(response, 422, "validation_error")["details"]["errors"][0]
    assert error["field"] == "body"
    assert error["type"] == "json_invalid"


def test_deeply_nested_body(trained_client: TestClient) -> None:
    response = trained_client.post("/predict", content="[" * 5000 + "]" * 5000, headers=JSON_HEADERS)
    assert_error(response, 422, "validation_error")


# --- модель не подходит ---

def test_feature_mismatch(trained_client: TestClient, churn_split: DataSplit) -> None:
    # модель обучена на другом наборе признаков, как будто списки признаков поменяли после обучения
    X_train, _, y_train, _ = churn_split
    other = create_pipeline(TrainingConfigChurn()).fit(X_train.assign(extra=1), y_train)
    service = trained_client.app.state.model_service
    service.model = service.model.model_copy(update={"pipeline": other})

    body = assert_error(trained_client.post("/predict", json=RISKY_CLIENT), 503, "feature_mismatch")
    assert body["details"]["expected"][-1] == "extra"


def test_model_failure(trained_client: TestClient, churn_split: DataSplit) -> None:
    # модель без predict_proba
    X_train, _, y_train, _ = churn_split
    broken = Pipeline([("preprocessor", create_preprocessor()), ("model", SVC())]).fit(X_train, y_train)
    service = trained_client.app.state.model_service
    service.model = service.model.model_copy(update={"pipeline": broken})

    body = assert_error(trained_client.post("/predict", json=RISKY_CLIENT), 500, "prediction_failed")
    assert body["details"]["clients"] == 1
