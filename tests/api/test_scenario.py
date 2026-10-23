"""Сквозной сценарий: обучение -> статус -> предсказание, и то же после перезапуска."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.main
from src.api.routes.prediction import LOYAL_CLIENT, RISKY_CLIENT


def test_train_status_predict(client: TestClient) -> None:
    # до обучения модели нет
    status = client.get("/model/status").json()
    assert status == {"trained": False, "model_type": None, "hyperparameters": None,
                      "trained_at": None, "metrics": None}
    response = client.post("/predict", json=RISKY_CLIENT)
    assert response.status_code == 503
    assert response.json()["code"] == "model_not_trained"

    # обучение
    response = client.post("/model/train", json={"model_type": "logreg"})
    assert response.status_code == 200
    metrics = response.json()
    assert set(metrics) == {"accuracy", "f1", "roc_auc"}
    assert all(0 <= value <= 1 for value in metrics.values())

    # статус показывает обученную модель с теми же метриками
    status = client.get("/model/status").json()
    assert status["trained"] is True
    assert status["model_type"] == "logreg"
    assert status["hyperparameters"] == {"max_iter": 1000}
    assert status["trained_at"] is not None
    assert status["metrics"] == metrics

    # предсказание для двух клиентов: ответ в том же порядке, рискованный клиент вероятнее уйдёт
    response = client.post("/predict", json=[RISKY_CLIENT, LOYAL_CLIENT])
    assert response.status_code == 200
    risky, loyal = response.json()["predictions"]
    for prediction in (risky, loyal):
        probabilities = prediction["probabilities"]
        assert set(probabilities) == {"0", "1"}
        assert probabilities["0"] + probabilities["1"] == pytest.approx(1, abs=1e-3)
        assert prediction["churn"] == (1 if probabilities["1"] > probabilities["0"] else 0)
    assert risky["probabilities"]["1"] > loyal["probabilities"]["1"]
    assert risky["churn"] == 1
    assert loyal["churn"] == 0


def test_model_loaded_after_restart(model_dir: Path) -> None:
    with TestClient(src.main.app) as client:
        client.post("/model/train", json={})
        status = client.get("/model/status").json()
        prediction = client.post("/predict", json=RISKY_CLIENT).json()

    assert (model_dir / "churn_model.joblib").exists()

    with TestClient(src.main.app) as restarted:
        assert restarted.get("/model/status").json() == status
        assert restarted.post("/predict", json=RISKY_CLIENT).json() == prediction
