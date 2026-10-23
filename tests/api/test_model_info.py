"""/model/status, /model/schema, /model/metrics, /model/compare."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.api.helpers import assert_error


def test_without_model_are_not_errors(client: TestClient) -> None:
    assert client.get("/model/status").json()["trained"] is False
    assert client.get("/model/schema").status_code == 200
    assert client.get("/model/metrics").json() == {"latest": None, "history": []}


# --- ошибки ---

@pytest.mark.parametrize(
    "params",
    [{"last": -1}, {"last": 101}, {"model_type": "xgboost"}],
    ids=["last_negative", "last_too_big", "unknown_model_type"],
)
def test_metrics_invalid_query(client: TestClient, params: dict[str, Any]) -> None:
    assert_error(client.get("/model/metrics", params=params), 422, "validation_error")


def test_compare_unknown_metric(client: TestClient) -> None:
    assert_error(client.get("/model/compare", params={"metric": "precision"}), 422, "validation_error")
