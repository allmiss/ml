from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from src.core.exceptions import InvalidHistoryFileError
from src.ml.history import TrainingHistory, compare_configs
from src.schemas.model import TrainingRecord, TrainMetrics

NOW = datetime(2026, 10, 1, 12, 0).astimezone()


def add(history: TrainingHistory, model_type: str = "logreg", f1: float = 0.3, **hyperparameters: Any) -> None:
    history.add(NOW, model_type, hyperparameters, 400, 100, TrainMetrics(accuracy=0.8, f1=f1, roc_auc=0.7))


def record(id: int, model_type: str, hyperparameters: dict[str, Any], f1: float, roc_auc: float) -> TrainingRecord:
    return TrainingRecord(
        id=id, trained_at=NOW, model_type=model_type, hyperparameters=hyperparameters,
        train_size=400, test_size=100, metrics=TrainMetrics(accuracy=0.8, f1=f1, roc_auc=roc_auc),
    )


# --- TrainingHistory ---

def test_empty_without_file(tmp_path: Path) -> None:
    history = TrainingHistory(tmp_path / "history.json")
    assert history.records == []
    assert not history.path.exists()


def test_add_numbers_records_and_saves(tmp_path: Path) -> None:
    history = TrainingHistory(tmp_path / "history.json")
    add(history)
    add(history, "random_forest")

    assert [r.id for r in history.records] == [1, 2]
    # новый объект читает то же из файла, нумерация продолжается
    reloaded = TrainingHistory(history.path)
    assert reloaded.records == history.records
    add(reloaded)
    assert reloaded.records[-1].id == 3


def test_last_newest_first_and_filtered(tmp_path: Path) -> None:
    history = TrainingHistory(tmp_path / "history.json")
    for model_type in ["logreg", "random_forest", "logreg", "random_forest", "logreg"]:
        add(history, model_type)

    assert [r.id for r in history.last(3)] == [5, 4, 3]
    assert [r.id for r in history.last(10, "random_forest")] == [4, 2]
    assert history.last(0) == []


@pytest.mark.parametrize(
    "content",
    ['{"not": "a list"}', "[{bad json", '[{"id": 1}]'],
    ids=["not_list", "bad_json", "bad_record"],
)
def test_damaged_file_is_not_overwritten(tmp_path: Path, content: str) -> None:
    path = tmp_path / "history.json"
    path.write_text(content, encoding="utf-8")
    history = TrainingHistory(path)

    with pytest.raises(InvalidHistoryFileError):
        add(history)
    assert path.read_text(encoding="utf-8") == content


# --- compare_configs ---

def test_compare_sorts_by_metric() -> None:
    records = [
        record(1, "logreg", {}, f1=0.05, roc_auc=0.61),
        record(2, "logreg", {"class_weight": "balanced"}, f1=0.35, roc_auc=0.60),
        record(3, "random_forest", {}, f1=0.17, roc_auc=0.65),
    ]
    assert [c.last_record_id for c in compare_configs(records, "f1")] == [2, 3, 1]
    assert [c.last_record_id for c in compare_configs(records, "roc_auc")] == [3, 1, 2]


def test_compare_merges_same_config() -> None:
    # одна и та же настройка дважды: одна строка, runs=2, метрики последнего запуска
    records = [
        record(1, "logreg", {"C": 1, "max_iter": 1000}, f1=0.30, roc_auc=0.6),
        record(2, "random_forest", {}, f1=0.10, roc_auc=0.6),
        record(3, "logreg", {"max_iter": 1000, "C": 1}, f1=0.32, roc_auc=0.6),
    ]
    configs = compare_configs(records, "f1")

    assert len(configs) == 2
    assert (configs[0].runs, configs[0].last_record_id, configs[0].metrics.f1) == (2, 3, 0.32)


def test_compare_empty() -> None:
    assert compare_configs([], "f1") == []
