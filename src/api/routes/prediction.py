from typing import Annotated

from fastapi import APIRouter, Body

from src.api.dependencies import ModelServiceDep
from src.api.routes.error_examples import PREDICT_ERRORS
from src.schemas.churn import FeatureVectorsChurn, PredictionResponseChurn

router = APIRouter(tags=["prediction"])

RISKY_CLIENT = {
    "monthly_fee": 29.99,
    "usage_hours": 3.5,
    "support_requests": 4,
    "account_age_months": 2,
    "failed_payments": 2,
    "region": "asia",
    "device_type": "mobile",
    "payment_method": "crypto",
    "autopay_enabled": 0,
}

LOYAL_CLIENT = {
    "monthly_fee": 29.99,
    "usage_hours": 40.0,
    "support_requests": 0,
    "account_age_months": 36,
    "failed_payments": 0,
    "region": "asia",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1,
}

PREDICT_EXAMPLES = {
    "one": {
        "summary": "Один клиент",
        "description": "Новый клиент с неуспешными платежами и без автоплатежа: высокий риск ухода.",
        "value": RISKY_CLIENT,
    },
    "batch": {
        "summary": "Несколько клиентов",
        "description": "Список клиентов: предсказания вернутся в том же порядке.",
        "value": [RISKY_CLIENT, LOYAL_CLIENT],
    },
}


@router.post(
    "/predict",
    summary="Предсказать отток клиентов",
    responses=PREDICT_ERRORS,
)
def predict(
    service: ModelServiceDep,
    features: Annotated[FeatureVectorsChurn, Body(openapi_examples=PREDICT_EXAMPLES)],
) -> PredictionResponseChurn:
    """Принимает одного клиента или список клиентов и возвращает для каждого
    предсказанный класс `churn` (1 — уйдёт, 0 — останется) и вероятности обоих классов.
    Клиент должен содержать ровно признаки из `GET /model/schema` с подходящими типами,
    иначе ответ 422 с ошибкой для каждого неверного поля."""
    return service.predict(features)
