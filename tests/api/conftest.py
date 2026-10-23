from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.main
from tests.api.helpers import ServiceStarter


@pytest.fixture
def model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, churn_csv: Path) -> Path:
    """Сервис работает на синтетическом датасете, а модель и история обучений лежат
    во временной папке, чтобы тесты не зависели от src/data и не трогали models/."""
    monkeypatch.setattr(src.main, "DATASET_PATH", churn_csv)
    monkeypatch.setattr(src.main, "MODEL_PATH", tmp_path / "churn_model.joblib")
    monkeypatch.setattr(src.main, "HISTORY_PATH", tmp_path / "training_history.json")
    return tmp_path


@pytest.fixture
def client(model_dir: Path) -> Iterator[TestClient]:
    # with запускает lifespan: создаются сервисы и загружается модель, как при старте сервера
    with TestClient(src.main.app) as client:
        yield client


@pytest.fixture
def trained_client(client: TestClient) -> TestClient:
    assert client.post("/model/train", json={}).status_code == 200
    return client


@pytest.fixture
def start_service(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> ServiceStarter:
    """Запуск сервиса на другом датасете: with start_service(path) as client: ...

    model_dir нужен ради подмены путей модели и истории на временную папку."""
    def start(dataset_path: Path) -> TestClient:
        monkeypatch.setattr(src.main, "DATASET_PATH", dataset_path)
        return TestClient(src.main.app)
    return start
