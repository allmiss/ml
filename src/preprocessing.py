import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE, TEST_SIZE
from src.exceptions import DataPreparationError
from src.models import ClassDistribution, FeatureVectorChurn

TARGET = "churn"

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# X_train, X_test, y_train, y_test
DataSplit = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def features_to_frame(features: list[FeatureVectorChurn]) -> pd.DataFrame:
    return pd.DataFrame([item.model_dump() for item in features], columns=FEATURES)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [column for column in FEATURES + [TARGET] if column not in df.columns]
    if missing:
        raise DataPreparationError("В данных нет нужных колонок", details={"missing": missing})

    df = df.dropna(subset=[TARGET])

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(X[NUMERIC_FEATURES].median())
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].fillna(X[CATEGORICAL_FEATURES].mode().iloc[0])

    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> DataSplit:
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError as e:
        raise DataPreparationError(
            "Не удалось разбить данные на обучающую и тестовую выборки",
            details={"rows": len(X), "test_size": test_size, "reason": str(e)},
        ) from None


def class_distribution(y: pd.Series) -> ClassDistribution:
    return ClassDistribution(
        counts=y.value_counts().sort_index().to_dict(),
        shares=y.value_counts(normalize=True).sort_index().round(3).to_dict(),
    )


def create_numeric_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
    ])


def create_categorical_pipeline() -> Pipeline:
    return Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])


def create_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", create_numeric_pipeline(), NUMERIC_FEATURES),
        ("cat", create_categorical_pipeline(), CATEGORICAL_FEATURES),
    ])


def fitted_categories(pipeline: Pipeline) -> dict[str, list[str]]:
    categorical = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    encoder = categorical[-1] if isinstance(categorical, Pipeline) else categorical
    return {
        name: [str(value) for value in values]
        for name, values in zip(encoder.feature_names_in_, encoder.categories_)
    }
