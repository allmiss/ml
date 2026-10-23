import csv
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.models import DatasetRowChurn, FeatureVectorChurn
from tests.api.helpers import ServiceStarter, assert_error


def test_info_reads_churn_dataset(client: TestClient, churn_csv: Path) -> None:
    with open(churn_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    response = client.get("/dataset/info")

    assert response.status_code == 200
    info = response.json()
    assert info["rows"] == len(rows)
    assert info["features"] == list(FeatureVectorChurn.model_fields)
    assert sum(info["churn_distribution"].values()) == len(rows)


def test_preview_matches_csv(client: TestClient, churn_csv: Path) -> None:
    with open(churn_csv, newline="", encoding="utf-8-sig") as f:
        first_rows = [row for _, row in zip(range(3), csv.DictReader(f))]

    response = client.get("/dataset/preview", params={"n": 3})

    assert response.status_code == 200
    assert response.json() == [DatasetRowChurn(**row).model_dump() for row in first_rows]


# --- ошибки ---

def test_not_found(start_service: ServiceStarter, tmp_path: Path) -> None:
    with start_service(tmp_path / "missing.csv") as client:
        body = assert_error(client.get("/dataset/info"), 503, "dataset_unavailable")
        assert body["details"] == {"file": "missing.csv"}
        assert_error(client.post("/model/train", json={}), 503, "dataset_unavailable")


def test_empty(start_service: ServiceStarter, tmp_path: Path, churn_frame: pd.DataFrame) -> None:
    path = tmp_path / "churn_dataset.csv"
    churn_frame.head(0).to_csv(path, index=False)
    with start_service(path) as client:
        body = assert_error(client.post("/model/train", json={}), 503, "dataset_unavailable")
        assert body["message"] == "Датасет пуст"


def test_invalid_row(start_service: ServiceStarter, tmp_path: Path, churn_frame: pd.DataFrame) -> None:
    path = tmp_path / "churn_dataset.csv"
    churn_frame.head(10).to_csv(path, index=False)
    lines = path.read_text(encoding="utf-8").splitlines()
    # в пятой строке файла (четвёртый клиент) вместо monthly_fee — текст
    lines[4] = "abc" + lines[4][lines[4].index(","):]
    path.write_text("\n".join(lines), encoding="utf-8")
    with start_service(path) as client:
        body = assert_error(client.post("/model/train", json={}), 500, "data_preparation_failed")
        assert body["details"]["line"] == 5
        assert body["details"]["errors"][0]["field"] == "monthly_fee"


def test_too_small_to_split(start_service: ServiceStarter, tmp_path: Path, churn_frame: pd.DataFrame) -> None:
    path = tmp_path / "churn_dataset.csv"
    churn_frame.head(3).assign(churn=[0, 0, 1]).to_csv(path, index=False)
    with start_service(path) as client:
        assert_error(client.post("/model/train", json={}), 500, "data_preparation_failed")
        assert_error(client.get("/dataset/split-info"), 500, "data_preparation_failed")
