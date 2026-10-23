"""Ошибки, общие для всего сервиса: неизвестный путь, метод, необработанное исключение."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException
from starlette.requests import Request

import src.main
from src.api.exception_handlers import handle_http_error
from src.services.model_service import ModelService
from tests.api.helpers import assert_error


def test_root(client: TestClient) -> None:
    assert client.get("/").json() == {"message": "ml churn service is running"}


def test_not_found(client: TestClient) -> None:
    assert_error(client.get("/nope"), 404, "not_found")


def test_method_not_allowed(client: TestClient) -> None:
    response = client.get("/predict")
    assert_error(response, 405, "method_not_allowed")
    assert response.headers["allow"] == "POST"


def test_http_exception_with_structured_detail() -> None:
    # в сервисе таких HTTPException нет, но обработчик не должен терять detail, если они появятся
    request = Request({"type": "http", "method": "GET", "path": "/x", "headers": []})
    response = asyncio.run(handle_http_error(request, HTTPException(409, detail={"reason": "conflict"})))
    assert response.status_code == 409
    assert json.loads(response.body) == {
        "code": "conflict", "message": "Conflict", "details": {"detail": {"reason": "conflict"}},
    }


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
