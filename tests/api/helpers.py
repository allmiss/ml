from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from httpx2 import Response

# start_service(путь к датасету) -> TestClient
ServiceStarter = Callable[[Path], TestClient]


def assert_error(response: Response, status_code: int, code: str) -> dict[str, Any]:
    """Ответ — ошибка в общем формате ErrorResponse с нужными кодами; возвращает тело."""
    assert response.status_code == status_code
    body = response.json()
    assert set(body) == {"code", "message", "details"}
    assert body["code"] == code
    assert body["message"]
    return body
