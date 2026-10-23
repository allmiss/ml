from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.main
from src.routes.prediction import RISKY_CLIENT
from tests.api.helpers import assert_error


def test_config_shown_in_status(client: TestClient) -> None:
    config = {"model_type": "random_forest", "hyperparameters": {"n_estimators": 50, "max_depth": 6}}
    response = client.post("/model/train", json=config)
    assert response.status_code == 200

    status = client.get("/model/status").json()
    assert status["model_type"] == "random_forest"
    assert status["hyperparameters"] == {"random_state": 42, "n_estimators": 50, "max_depth": 6}
    assert client.post("/predict", json=RISKY_CLIENT).status_code == 200


def test_is_reproducible(client: TestClient) -> None:
    first = client.post("/model/train", json={}).json()
    second = client.post("/model/train", json={}).json()
    assert first == second


def test_recorded_in_history(client: TestClient) -> None:
    metrics = client.post("/model/train", json={}).json()
    history = client.get("/model/metrics", params={"last": 5}).json()
    assert history["latest"]["metrics"] == metrics
    assert len(history["history"]) == 1


# --- ошибки ---

@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"model_type": "xgboost"}, "Неизвестный тип модели 'xgboost'"),
        ({"model_type": "logreg", "hyperparameters": {"foo": 1}}, "Неверные гиперпараметры для logreg"),
        ({"model_type": "logreg", "hyperparameters": {"C": -1}}, "Не удалось обучить logreg"),
    ],
    ids=["unknown_model_type", "unknown_hyperparameter", "invalid_value"],
)
def test_invalid_config(trained_client: TestClient, config: dict[str, Any], message: str) -> None:
    before = trained_client.get("/model/status").json()

    body = assert_error(trained_client.post("/model/train", json=config), 400, "invalid_training_config")

    assert body["message"] == message
    # прошлая модель и история не меняются
    assert trained_client.get("/model/status").json() == before
    assert len(trained_client.get("/model/metrics", params={"last": 10}).json()["history"]) == 1


@pytest.mark.parametrize(
    "config",
    [{"model_type": 5}, {"hyperparameters": [1]}],
    ids=["model_type_not_string", "hyperparameters_not_dict"],
)
def test_config_wrong_types(client: TestClient, config: dict[str, Any]) -> None:
    assert_error(client.post("/model/train", json=config), 422, "validation_error")


def test_model_cannot_be_saved(model_dir: Path) -> None:
    # на месте файла модели папка — сохранить модель нельзя
    (model_dir / "churn_model.joblib").mkdir()
    with TestClient(src.main.app, raise_server_exceptions=False) as client:
        assert_error(client.post("/model/train", json={}), 500, "internal_error")
        assert client.get("/model/status").json()["trained"] is False


def test_damaged_history_file(model_dir: Path) -> None:
    history_path = model_dir / "training_history.json"
    history_path.write_text('[{"broken": 1}]', encoding="utf-8")

    with TestClient(src.main.app) as client:
        assert_error(client.get("/model/metrics"), 500, "invalid_history_file")
        # обучение падает до сохранения модели, файл истории не перезаписывается
        assert_error(client.post("/model/train", json={}), 500, "invalid_history_file")

    assert not (model_dir / "churn_model.joblib").exists()
    assert history_path.read_text(encoding="utf-8") == '[{"broken": 1}]'
