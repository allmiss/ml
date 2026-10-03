import csv
from pathlib import Path

from src.models import DatasetRowChurn

DATASET_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"


def load_dataset(path: Path = DATASET_PATH) -> list[DatasetRowChurn]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [DatasetRowChurn(**row) for row in csv.DictReader(f)]


def to_records(rows: list[DatasetRowChurn]) -> list[dict]:
    return [row.model_dump() for row in rows]
