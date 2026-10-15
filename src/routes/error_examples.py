"""Примеры ошибок для /docs.

Каждый пример — реальный ответ сервиса в формате ErrorResponse. Примеры
сгруппированы по HTTP-коду; в /docs у кода ответа появляется список примеров,
между которыми можно переключаться.
"""

from typing import Any

from src.models import ErrorResponse


def error_doc(description: str, examples: dict[str, tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Описание одного кода ответа: examples — {имя: (заголовок, тело ответа)}."""
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {
            "application/json": {
                "examples": {name: {"summary": summary, "value": value} for name, (summary, value) in examples.items()},
            },
        },
    }


def validation_error(*errors: dict[str, Any]) -> dict[str, Any]:
    return {"code": "validation_error", "message": "Запрос не прошёл проверку", "details": {"errors": list(errors)}}


INVALID_JSON = (
    "Тело запроса не JSON",
    validation_error({
        "field": "body",
        "type": "json_invalid",
        "message": "Тело запроса не JSON: Expecting property name enclosed in double quotes",
    }),
)

DATASET_NOT_FOUND = (
    "Файл датасета не найден",
    {
        "code": "dataset_unavailable",
        "message": "Датасет не загружен: файл churn_dataset.csv не найден",
        "details": {"file": "churn_dataset.csv"},
    },
)

DATASET_EMPTY = (
    "В датасете нет строк",
    {"code": "dataset_unavailable", "message": "Датасет пуст", "details": {"file": "churn_dataset.csv"}},
)

TRAIN_ERRORS: dict[int | str, dict[str, Any]] = {
    400: error_doc("Неподходящая конфигурация обучения, сохранённая модель не меняется", {
        "unknown_model_type": ("Неизвестный тип модели", {
            "code": "invalid_training_config",
            "message": "Неизвестный тип модели 'xgboost'",
            "details": {"model_type": "xgboost", "supported": ["logreg", "random_forest"]},
        }),
        "unknown_hyperparameter": ("У модели нет такого гиперпараметра", {
            "code": "invalid_training_config",
            "message": "Неверные гиперпараметры для logreg",
            "details": {
                "model_type": "logreg",
                "reason": "LogisticRegression.__init__() got an unexpected keyword argument 'foo'",
            },
        }),
        "invalid_hyperparameter_value": ("Недопустимое значение гиперпараметра (C=-1)", {
            "code": "invalid_training_config",
            "message": "Не удалось обучить logreg",
            "details": {
                "model_type": "logreg",
                "reason": "The 'C' parameter of LogisticRegression must be a float in the range (0.0, inf]. "
                          "Got -1 instead.",
            },
        }),
    }),
    422: error_doc("Тело запроса не соответствует TrainingConfigChurn", {
        "wrong_types": ("Неверные типы полей", validation_error(
            {"field": "model_type", "type": "string_type", "message": "Input should be a valid string", "input": 5},
            {"field": "hyperparameters", "type": "dict_type", "message": "Input should be a valid dictionary",
             "input": "list"},
        )),
        "invalid_json": INVALID_JSON,
    }),
    500: error_doc("Данные датасета не удалось подготовить к обучению", {
        "invalid_row": ("Неверное значение в строке CSV", {
            "code": "data_preparation_failed",
            "message": "Неверные данные в строке 5 файла churn_dataset.csv",
            "details": {
                "file": "churn_dataset.csv",
                "line": 5,
                "errors": [{
                    "field": "monthly_fee",
                    "message": "Input should be a valid number, unable to parse string as a number",
                }],
            },
        }),
        "split_failed": ("Слишком мало данных для разбиения на выборки", {
            "code": "data_preparation_failed",
            "message": "Не удалось разбить данные на обучающую и тестовую выборки",
            "details": {
                "rows": 3,
                "test_size": 0.2,
                "reason": "The least populated classes in y have only 1 member, which is too few. "
                          "The minimum number of groups for any class cannot be less than 2. "
                          "Classes with too few members are: [1]",
            },
        }),
    }),
    503: error_doc("Датасет недоступен", {
        "dataset_not_found": DATASET_NOT_FOUND,
        "dataset_empty": DATASET_EMPTY,
    }),
}

PREDICT_ERRORS: dict[int | str, dict[str, Any]] = {
    422: error_doc("Неверный набор признаков или типы значений", {
        "missing_features": ("Не хватает признаков", validation_error(
            {"field": "0.usage_hours", "type": "missing", "message": "Field required"},
            {"field": "0.region", "type": "missing", "message": "Field required"},
        )),
        "extra_feature": ("Лишний признак", validation_error(
            {"field": "0.extra", "type": "extra_forbidden", "message": "Extra inputs are not permitted", "input": 1},
        )),
        "wrong_types": ("Неверные типы значений", validation_error(
            {"field": "0.monthly_fee", "type": "float_parsing",
             "message": "Input should be a valid number, unable to parse string as a number", "input": "abc"},
            {"field": "0.support_requests", "type": "int_from_float",
             "message": "Input should be a valid integer, got a number with a fractional part", "input": 1.5},
            {"field": "0.region", "type": "string_type", "message": "Input should be a valid string", "input": 5},
        )),
        "not_finite": ("NaN или Infinity вместо числа", validation_error(
            {"field": "0.monthly_fee", "type": "finite_number", "message": "Input should be a finite number",
             "input": "nan"},
        )),
        "error_in_batch": ("Ошибка у второго клиента в списке", validation_error(
            {"field": "1.usage_hours", "type": "float_parsing",
             "message": "Input should be a valid number, unable to parse string as a number", "input": "x"},
        )),
        "invalid_json": INVALID_JSON,
    }),
    500: error_doc("Модель не смогла посчитать предсказание", {
        "prediction_failed": ("Модель не умеет считать вероятности", {
            "code": "prediction_failed",
            "message": "Модель не смогла посчитать предсказание",
            "details": {"clients": 1, "reason": "This 'Pipeline' has no attribute 'predict_proba'"},
        }),
    }),
    503: error_doc("Модели нет или она не подходит к текущим признакам", {
        "model_not_trained": ("Модель не обучена", {
            "code": "model_not_trained",
            "message": "Модель не обучена, обучите её через POST /model/train",
            "details": None,
        }),
        "feature_mismatch": ("Модель обучена на других признаках", {
            "code": "feature_mismatch",
            "message": "Признаки сервиса не совпадают с признаками модели, переобучите её через POST /model/train",
            "details": {
                "expected": ["monthly_fee", "usage_hours", "support_requests", "account_age_months",
                             "failed_payments", "autopay_enabled", "region", "device_type"],
                "received": ["monthly_fee", "usage_hours", "support_requests", "account_age_months",
                             "failed_payments", "autopay_enabled", "region", "device_type", "payment_method"],
            },
        }),
    }),
}
