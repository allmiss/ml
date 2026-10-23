import pandas as pd
import pytest

from src.ml.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    create_preprocessor,
    fitted_categories,
    prepare_data,
)
from src.ml.training import create_pipeline
from src.schemas.model import TrainingConfigChurn


def test_output_columns(small_frame: pd.DataFrame) -> None:
    X, _ = prepare_data(small_frame)
    transformed = create_preprocessor().fit_transform(X)
    # числовые как есть + по колонке на каждое значение категорий (region: asia, europe)
    expected = len(NUMERIC_FEATURES) + sum(X[c].nunique() for c in CATEGORICAL_FEATURES)
    assert transformed.shape == (len(X), expected)


def test_scales_numeric(small_frame: pd.DataFrame) -> None:
    X, _ = prepare_data(small_frame)
    transformed = create_preprocessor().fit_transform(X)
    numeric = transformed[:, : len(NUMERIC_FEATURES)]
    assert numeric.mean(axis=0) == pytest.approx(0, abs=1e-9)


def test_ignores_unknown_category(small_frame: pd.DataFrame) -> None:
    X, _ = prepare_data(small_frame)
    preprocessor = create_preprocessor().fit(X)
    unknown = X.head(1).assign(region="mars")
    assert preprocessor.transform(unknown).shape[1] == preprocessor.transform(X.head(1)).shape[1]


def test_fitted_categories(small_frame: pd.DataFrame) -> None:
    X, y = prepare_data(small_frame.assign(churn=[0, 1, 0, 1, 0]))
    pipeline = create_pipeline(TrainingConfigChurn()).fit(X, y)
    categories = fitted_categories(pipeline)
    assert set(categories) == set(CATEGORICAL_FEATURES)
    assert categories["region"] == ["asia", "europe"]
