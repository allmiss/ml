import pandas as pd
from sklearn.model_selection import train_test_split

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


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = df.dropna(subset=[TARGET])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int)

    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(X[NUMERIC_FEATURES].median())
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].fillna(X[CATEGORICAL_FEATURES].mode().iloc[0])

    return X, y, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def churn_distribution(y_train: pd.Series, y_test: pd.Series) -> dict:
    return {
        name: {
            "counts": y.value_counts().sort_index().to_dict(),
            "shares": y.value_counts(normalize=True).sort_index().round(3).to_dict(),
        }
        for name, y in (("train", y_train), ("test", y_test))
    }
