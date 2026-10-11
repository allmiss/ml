import csv
from pathlib import Path

import pandas as pd

from src.exceptions import DatasetUnavailableError
from src.models import DatasetRowChurn


class ChurnDataset:
    def __init__(self, path: Path):
        self.path = path
        self._rows: list[DatasetRowChurn] = []

    @property
    def rows(self) -> list[DatasetRowChurn]:
        if not self._rows:
            self._rows = self._load()
        return self._rows

    def to_records(self) -> list[dict]:
        return [row.model_dump() for row in self.rows]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_records())

    def _load(self) -> list[DatasetRowChurn]:
        if not self.path.exists():
            raise DatasetUnavailableError(f"Датасет не загружен: файл {self.path.name} не найден")
        with open(self.path, newline="", encoding="utf-8-sig") as f:
            rows = [DatasetRowChurn(**row) for row in csv.DictReader(f)]
        if not rows:
            raise DatasetUnavailableError("Датасет пуст")
        return rows
