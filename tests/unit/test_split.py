import pandas as pd
import pytest

from src.exceptions import DataPreparationError
from src.preprocessing import class_distribution, prepare_data, split_data


def test_sizes_and_stratification(churn_frame: pd.DataFrame) -> None:
    X, y = prepare_data(churn_frame)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_test) == round(len(X) * 0.2)
    # доля ушедших в обеих выборках почти как во всём датасете
    assert y_train.mean() == pytest.approx(y.mean(), abs=0.01)
    assert y_test.mean() == pytest.approx(y.mean(), abs=0.01)


def test_is_reproducible(churn_frame: pd.DataFrame) -> None:
    X, y = prepare_data(churn_frame)
    first = split_data(X, y)
    second = split_data(X, y)
    for a, b in zip(first, second):
        assert list(a.index) == list(b.index)


def test_fails_when_class_has_one_client(small_frame: pd.DataFrame) -> None:
    # в классе 1 всего один клиент — его нельзя положить и в обучающую, и в тестовую выборку
    X, y = prepare_data(small_frame.assign(churn=[0, 0, 0, 1, 0]))
    with pytest.raises(DataPreparationError, match="разбить"):
        split_data(X, y)


def test_class_distribution() -> None:
    distribution = class_distribution(pd.Series([0, 0, 0, 1]))
    assert distribution.counts == {0: 3, 1: 1}
    assert distribution.shares == {0: 0.75, 1: 0.25}
