from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.api.helpers import ServiceStarter


def test_without_model(client: TestClient, churn_frame: pd.DataFrame) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "model": {"available": False, "model_type": None, "trained_at": None},
        "dataset": {"loaded": True, "rows": len(churn_frame), "error": None},
    }


def test_ok_after_training(trained_client: TestClient) -> None:
    health = trained_client.get("/health").json()

    assert health["status"] == "ok"
    assert health["model"]["available"] is True
    assert health["model"]["model_type"] == "logreg"
    assert health["model"]["trained_at"] == trained_client.get("/model/status").json()["trained_at"]


def test_dataset_unavailable_is_not_an_error(start_service: ServiceStarter, tmp_path: Path) -> None:
    with start_service(tmp_path / "missing.csv") as client:
        response = client.get("/health")

    assert response.status_code == 200
    health = response.json()
    assert health["status"] == "degraded"
    assert health["dataset"] == {
        "loaded": False,
        "rows": None,
        "error": "Датасет не загружен: файл missing.csv не найден",
    }


@pytest.mark.usefixtures("trained_client")
def test_model_available_after_restart_without_dataset(start_service: ServiceStarter, tmp_path: Path) -> None:
    # модель уже сохранена; после перезапуска датасет пропал — модель всё равно доступна
    with start_service(tmp_path / "missing.csv") as client:
        health = client.get("/health").json()

    assert health["model"]["available"] is True
    assert health["dataset"]["loaded"] is False
    assert health["status"] == "degraded"
