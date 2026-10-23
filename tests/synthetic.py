"""Тестовые данные: строки и синтетический датасет churn с фиксированным seed."""

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing import FEATURES, TARGET

SEED = 42
DATASET_SIZE = 500


def make_row(**overrides: Any) -> dict[str, Any]:
    """Строка датасета с корректными значениями; нужные поля можно переопределить."""
    row = {
        "monthly_fee": 9.99,
        "usage_hours": 10.0,
        "support_requests": 1,
        "account_age_months": 12,
        "failed_payments": 0,
        "region": "asia",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 1,
        "churn": 0,
    }
    return {**row, **overrides}


def make_churn_frame(n: int = DATASET_SIZE, seed: int = SEED) -> pd.DataFrame:
    """Синтетический churn датасет: при одном seed всегда одни и те же строки.

    Уход клиента зависит от признаков так же по смыслу, как в настоящих данных:
    неуспешные платежи и отсутствие автоплатежа повышают шанс ухода, а частое
    использование и большой стаж — снижают. Тариф, обращения, регион, устройство
    и способ оплаты на уход не влияют.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "monthly_fee": rng.choice([9.99, 19.99, 29.99], n),
        "usage_hours": rng.uniform(0, 50, n).round(2),
        "support_requests": rng.poisson(1, n),
        "account_age_months": rng.integers(1, 61, n),
        "failed_payments": rng.poisson(0.4, n),
        "region": rng.choice(["africa", "america", "asia", "europe"], n),
        "device_type": rng.choice(["desktop", "mobile", "tablet"], n),
        "payment_method": rng.choice(["card", "crypto", "paypal"], n),
        "autopay_enabled": rng.integers(0, 2, n),
    })
    score = (
        0.5
        + 1.2 * frame["failed_payments"]
        - 1.0 * frame["autopay_enabled"]
        - 0.06 * frame["usage_hours"]
        - 0.03 * frame["account_age_months"]
    )
    frame["churn"] = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    return frame


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """CSV того же формата, что src/data/churn_dataset.csv; без строк — только заголовок."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else FEATURES + [TARGET])
        writer.writeheader()
        writer.writerows(rows)
