import logging
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.main
from src.api.routes.prediction import RISKY_CLIENT
from tests.api.helpers import ServiceStarter

# значение, которого нет ни в каком другом месте: по нему видно, попали ли признаки клиента в лог
MARKER = 37.123456


def messages(caplog: pytest.LogCaptureFixture, level: int = logging.DEBUG) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name.startswith("src.") and r.levelno >= level]


def test_dataset_loading(client: TestClient, caplog: pytest.LogCaptureFixture, churn_frame: pd.DataFrame) -> None:
    with caplog.at_level(logging.INFO, logger="src"):
        client.get("/dataset/info")
    assert any(f"{len(churn_frame)} строк" in m for m in messages(caplog))


def test_training(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="src"):
        metrics = client.post("/model/train", json={"model_type": "logreg"}).json()

    logged = " ".join(messages(caplog))
    assert "Обучение logreg {'max_iter': 1000}" in logged
    assert f"f1={metrics['f1']}" in logged
    assert "запись истории #1" in logged


def test_predict_logs_counts_not_client_data(trained_client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    client = {**RISKY_CLIENT, "usage_hours": MARKER}
    with caplog.at_level(logging.DEBUG, logger="src"):
        trained_client.post("/predict", json=[client, client])

    logged = " ".join(messages(caplog))
    assert "Предсказание для 2 клиентов" in logged
    assert str(MARKER) not in logged


def test_validation_error_does_not_log_values(trained_client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="src"):
        trained_client.post("/predict", json={**RISKY_CLIENT, "usage_hours": "x", "extra": MARKER})

    warnings = messages(caplog, logging.WARNING)
    assert warnings == [
        "POST /predict -> 422 validation_error: 0.usage_hours (float_parsing), 0.extra (extra_forbidden)"
    ]
    assert str(MARKER) not in " ".join(messages(caplog))


def test_error_levels(start_service: ServiceStarter, tmp_path: Path, caplog: pytest.LogCaptureFixture,
                      churn_frame: pd.DataFrame) -> None:
    # 503 «модель не обучена» — ожидаемое состояние, WARNING; сбой подготовки данных (500) — ERROR
    path = tmp_path / "churn_dataset.csv"
    churn_frame.head(3).assign(churn=[0, 0, 1]).to_csv(path, index=False)
    with start_service(path) as client, caplog.at_level(logging.INFO, logger="src"):
        client.post("/predict", json=RISKY_CLIENT)
        client.post("/model/train", json={})

    levels = {
        r.getMessage().split(" -> ")[1].split(":")[0]: r.levelname for r in caplog.records if " -> " in r.getMessage()
    }
    assert levels == {"503 model_not_trained": "WARNING", "500 data_preparation_failed": "ERROR"}


def test_unreadable_model_file_at_startup(model_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    (model_dir / "churn_model.joblib").mkdir()
    with caplog.at_level(logging.INFO, logger="src"), TestClient(src.main.app) as client:
        assert client.get("/model/status").json()["trained"] is False

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert [r.getMessage().split(" из ")[0] for r in errors] == ["Не удалось загрузить модель"]
    assert errors[0].exc_info is not None
    assert not any("неподдерживаемом формате" in m for m in messages(caplog))


def test_old_model_file_at_startup(model_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    (model_dir / "churn_model.joblib").write_bytes(b"not a model")
    with caplog.at_level(logging.INFO, logger="src"), TestClient(src.main.app):
        pass
    assert [(r.levelname, "переобучите" in r.getMessage()) for r in caplog.records if r.levelno >= logging.WARNING] == [
        ("WARNING", True)
    ]
