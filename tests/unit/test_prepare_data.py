import pandas as pd
import pytest

from src.exceptions import DataPreparationError
from src.preprocessing import FEATURES, TARGET, prepare_data


def test_drops_rows_without_target(small_frame: pd.DataFrame) -> None:
    X, y = prepare_data(small_frame)
    assert len(X) == len(y) == 4
    assert y.dtype == int


def test_fills_missing_values(small_frame: pd.DataFrame) -> None:
    X, _ = prepare_data(small_frame)
    assert not X.isna().any().any()
    # медиана monthly_fee среди строк с churn: 10, 20, 60 -> 20 (среднее было бы 30)
    assert X.loc[2, "monthly_fee"] == 20.0
    # мода region: asia
    assert X.loc[3, "region"] == "asia"


def test_keeps_feature_order(small_frame: pd.DataFrame) -> None:
    X, _ = prepare_data(small_frame[list(reversed(small_frame.columns))])
    assert list(X.columns) == FEATURES


def test_does_not_change_input(small_frame: pd.DataFrame) -> None:
    before = small_frame.copy()
    prepare_data(small_frame)
    pd.testing.assert_frame_equal(small_frame, before)


def test_missing_columns(small_frame: pd.DataFrame) -> None:
    with pytest.raises(DataPreparationError) as exc:
        prepare_data(small_frame.drop(columns=["region", TARGET]))
    assert exc.value.details["missing"] == ["region", TARGET]
