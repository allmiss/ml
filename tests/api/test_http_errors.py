"""Ошибки, общие для всего сервиса: неизвестный путь, метод, необработанное исключение."""

import pytest
from fastapi.testclient import TestClient

import src.main
from src.services.model_service import ModelService
from tests.api.helpers import assert_error


def test_not_found(client: TestClient) -> None:
    assert_error(client.get("/nope"), 404, "not_found")


def test_method_not_allowed(client: TestClient) -> None:
    response = client.get("/predict")
    assert_error(response, 405, "method_not_allowed")
    assert response.headers["allow"] == "POST"


@pytest.mark.usefixtures("model_dir")
def test_unexpected_error_hides_details(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: ModelService) -> None:
        raise RuntimeError("секретная деталь: пароль 12345")

    monkeypatch.setattr(ModelService, "status", fail)
    with TestClient(src.main.app, raise_server_exceptions=False) as client:
        response = client.get("/model/status")

    assert_error(response, 500, "internal_error")
    assert "12345" not in response.text
    assert "Traceback" not in response.text
