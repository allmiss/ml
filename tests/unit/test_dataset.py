from pathlib import Path

import pytest

from src.dataset import ChurnDataset
from src.exceptions import DataPreparationError, DatasetUnavailableError
from tests.synthetic import make_row, write_csv


def test_loads_rows(tmp_path: Path) -> None:
    path = tmp_path / "churn.csv"
    write_csv(path, [make_row(), make_row(churn=1)])
    frame = ChurnDataset(path).to_frame()
    assert len(frame) == 2
    assert list(frame.columns) == list(make_row())


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetUnavailableError, match="не найден"):
        ChurnDataset(tmp_path / "nope.csv").rows


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "churn.csv"
    write_csv(path, [])
    with pytest.raises(DatasetUnavailableError, match="пуст"):
        ChurnDataset(path).rows


def test_invalid_value_points_to_line(tmp_path: Path) -> None:
    path = tmp_path / "churn.csv"
    write_csv(path, [make_row(), make_row(monthly_fee="abc")])
    with pytest.raises(DataPreparationError) as exc:
        ChurnDataset(path).rows
    assert exc.value.details["line"] == 3
    assert exc.value.details["errors"][0]["field"] == "monthly_fee"
