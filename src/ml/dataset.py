import csv
import logging
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.core.exceptions import DataPreparationError, DatasetUnavailableError
from src.schemas.churn import DatasetRowChurn

logger = logging.getLogger(__name__)


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
            raise DatasetUnavailableError(
                f"Датасет не загружен: файл {self.path.name} не найден", details={"file": self.path.name}
            )
        with open(self.path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            try:
                rows = [DatasetRowChurn(**row) for row in reader]
            except ValidationError as e:
                raise DataPreparationError(
                    f"Неверные данные в строке {reader.line_num} файла {self.path.name}",
                    details={
                        "file": self.path.name,
                        "line": reader.line_num,
                        "errors": [
                            {"field": ".".join(map(str, err["loc"])), "message": err["msg"]} for err in e.errors()
                        ],
                    },
                ) from None
        if not rows:
            raise DatasetUnavailableError("Датасет пуст", details={"file": self.path.name})
        churned = sum(row.churn for row in rows)
        logger.info(
            "Датасет %s загружен: %d строк, ушедших %.1f%%", self.path.name, len(rows), 100 * churned / len(rows)
        )
        return rows
