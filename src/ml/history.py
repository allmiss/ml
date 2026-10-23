import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from src.core.exceptions import InvalidHistoryFileError
from src.schemas.model import ConfigComparison, MetricName, TrainingRecord, TrainMetrics

RECORDS = TypeAdapter(list[TrainingRecord])


class TrainingHistory:
    """История обучений в JSON-файле: список TrainingRecord от старых к новым.

    Файл читается один раз при первом обращении, дальше записи живут в памяти,
    а каждая новая запись сразу дописывается в файл.
    """

    def __init__(self, path: Path):
        self.path = path
        self._records: list[TrainingRecord] | None = None

    @property
    def records(self) -> list[TrainingRecord]:
        if self._records is None:
            self._records = self._load()
        return self._records

    def last(self, n: int, model_type: str | None = None) -> list[TrainingRecord]:
        """n последних записей, новые первыми; с model_type — только записи этого типа модели."""
        records = [r for r in reversed(self.records) if model_type is None or r.model_type == model_type]
        return records[:n]

    def add(
        self,
        trained_at: datetime,
        model_type: str,
        hyperparameters: dict[str, Any],
        train_size: int,
        test_size: int,
        metrics: TrainMetrics,
    ) -> TrainingRecord:
        records = self.records
        record = TrainingRecord(
            id=records[-1].id + 1 if records else 1,
            trained_at=trained_at,
            model_type=model_type,
            hyperparameters=hyperparameters,
            train_size=train_size,
            test_size=test_size,
            metrics=metrics,
        )
        self._save([*records, record])
        records.append(record)
        return record

    def _load(self) -> list[TrainingRecord]:
        if not self.path.exists():
            return []
        try:
            return RECORDS.validate_json(self.path.read_bytes())
        except ValidationError:
            raise InvalidHistoryFileError(
                f"Файл истории обучений {self.path.name} повреждён", details={"file": self.path.name}
            ) from None

    def _save(self, records: list[TrainingRecord]) -> None:
        # Пишем во временный файл и подменяем им старый, чтобы при сбое
        # посреди записи не остаться с наполовину записанной историей
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(RECORDS.dump_json(records, indent=2))
        tmp.replace(self.path)


def compare_configs(records: list[TrainingRecord], metric: MetricName) -> list[ConfigComparison]:
    """Сводит записи истории по настройкам (тип модели + гиперпараметры) и сортирует
    их по метрике, лучшие первыми. Повторные обучения с той же настройкой не
    дублируются: берутся метрики последнего из них, а в runs — сколько их было."""
    configs: dict[str, ConfigComparison] = {}
    for record in records:
        key = json.dumps([record.model_type, record.hyperparameters], sort_keys=True, default=str)
        runs = configs[key].runs + 1 if key in configs else 1
        configs[key] = ConfigComparison(
            model_type=record.model_type,
            hyperparameters=record.hyperparameters,
            metrics=record.metrics,
            runs=runs,
            last_record_id=record.id,
            last_trained_at=record.trained_at,
        )

    return sorted(configs.values(), key=lambda config: getattr(config.metrics, metric), reverse=True)
