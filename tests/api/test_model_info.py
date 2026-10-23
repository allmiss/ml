"""/model/status, /model/schema, /model/metrics, /model/compare."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.api.helpers import assert_error


def test_without_model_are_not_errors(client: TestClient) -> None:
    assert client.get("/model/status").json()["trained"] is False
    assert client.get("/model/schema").status_code == 200
    assert client.get("/model/metrics").json() == {"latest": None, "history": []}


def test_compare_ranks_trained_configs(client: TestClient) -> None:
    configs = [
        {"model_type": "logreg"},
        {"model_type": "logreg", "hyperparameters": {"class_weight": "balanced"}},
        {"model_type": "logreg"},
    ]
    for config in configs:
        assert client.post("/model/train", json=config).status_code == 200

    compare = client.get("/model/compare", params={"metric": "f1"}).json()

    assert compare["metric"] == "f1"
    assert len(compare["configs"]) == 2
    f1 = [c["metrics"]["f1"] for c in compare["configs"]]
    assert f1 == sorted(f1, reverse=True)
    assert compare["best"] == compare["configs"][0]
    assert {c["hyperparameters"].get("class_weight"): c["runs"] for c in compare["configs"]} == {None: 2, "balanced": 1}


def test_compare_filtered_by_model_type(client: TestClient) -> None:
    client.post("/model/train", json={"model_type": "logreg"})
    compare = client.get("/model/compare", params={"model_type": "random_forest"}).json()
    assert compare == {"metric": "f1", "best": None, "configs": []}


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
