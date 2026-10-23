"""Признаки клиента, строка датасета и ответ /predict."""

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


def _wrap_single_client(value: Any) -> Any:
    return [value] if isinstance(value, dict) else value


# Тело /predict: один клиент или список клиентов. Один клиент сразу
# заворачивается в список, поэтому проверка идёт по одной схеме и ошибки
# указывают на конкретное поле, а не на обе ветки «объект | список».
FeatureVectorsChurn = Annotated[
    list[FeatureVectorChurn],
    BeforeValidator(_wrap_single_client, json_schema_input_type=FeatureVectorChurn | list[FeatureVectorChurn]),
]


class ChurnPrediction(BaseModel):
    churn: int
    probabilities: dict[int, float]


class PredictionResponseChurn(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "predictions": [
                {"churn": 1, "probabilities": {"0": 0.3278, "1": 0.6722}},
                {"churn": 0, "probabilities": {"0": 0.9807, "1": 0.0193}},
            ],
        }],
    })

    predictions: list[ChurnPrediction]
