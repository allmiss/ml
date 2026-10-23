import pandas as pd

from src.models import DatasetRowChurn
from src.preprocessing import TARGET
from tests.synthetic import make_churn_frame


def test_synthetic_dataset_is_reproducible() -> None:
    pd.testing.assert_frame_equal(make_churn_frame(), make_churn_frame())
    assert not make_churn_frame(seed=1).equals(make_churn_frame())


def test_synthetic_dataset_is_valid(churn_frame: pd.DataFrame) -> None:
    # строки проходят ту же проверку, что и строки настоящего CSV
    rows = [DatasetRowChurn(**row) for row in churn_frame.to_dict("records")]
    assert len(rows) == len(churn_frame)
    assert 0.1 < churn_frame[TARGET].mean() < 0.3
