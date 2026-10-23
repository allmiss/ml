from pathlib import Path

import pandas as pd
import pytest

from src.ml.preprocessing import DataSplit, prepare_data, split_data
from tests.synthetic import make_churn_frame, make_row


@pytest.fixture
def small_frame() -> pd.DataFrame:
    """Маленький датасет с пропусками: по одному в числовом и категориальном признаке и в churn."""
    return pd.DataFrame([
        make_row(monthly_fee=10.0, region="asia"),
        make_row(monthly_fee=20.0, region="europe"),
        make_row(monthly_fee=None, region="asia"),
        make_row(monthly_fee=60.0, region=None),
        make_row(monthly_fee=40.0, churn=None),
    ])


@pytest.fixture(scope="session")
def churn_frame() -> pd.DataFrame:
    return make_churn_frame()


@pytest.fixture(scope="session")
def churn_split(churn_frame: pd.DataFrame) -> DataSplit:
    return split_data(*prepare_data(churn_frame))


@pytest.fixture(scope="session")
def churn_csv(churn_frame: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Синтетический датасет, записанный в CSV того же формата, что src/data/churn_dataset.csv."""
    path = tmp_path_factory.mktemp("data") / "churn_dataset.csv"
    churn_frame.to_csv(path, index=False)
    return path
