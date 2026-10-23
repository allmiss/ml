import logging
import math
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from src.exceptions import ChurnServiceError
from src.models import ErrorResponse

logger = logging.getLogger(__name__)


def error_response(status_code: int, error: ErrorResponse, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error.model_dump(mode="json"), headers=headers)


async def handle_service_error(request: Request, exc: ChurnServiceError) -> JSONResponse:
    # ERROR — только сбои сервиса (500); 4xx и 503 (нет модели или датасета) — ожидаемые ситуации
    level = logging.ERROR if exc.status_code == 500 else logging.WARNING
    details = f" {exc.details}" if exc.details else ""
    logger.log(
        level, "%s %s -> %d %s: %s%s",
        request.method, request.url.path, exc.status_code, exc.code, exc.message, details,
    )
    return error_response(exc.status_code, ErrorResponse(code=exc.code, message=exc.message, details=exc.details))


async def handle_http_error(request: Request, exc: HTTPException) -> JSONResponse:
    status = HTTPStatus(exc.status_code)
    logger.warning("%s %s -> %d %s", request.method, request.url.path, exc.status_code, status.name.lower())
    if isinstance(exc.detail, str):
        message, details = exc.detail, None
    else:
        message, details = status.phrase, {"detail": exc.detail}
    error = ErrorResponse(code=status.name.lower(), message=message, details=details)
    return error_response(exc.status_code, error, headers=exc.headers)


def input_for_response(value: Any) -> Any:
    """Переданное значение для details: только простые значения, которые
    точно сериализуются в JSON. NaN и Infinity — строкой, вложенные
    объекты и списки — только названием типа."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return type(value).__name__


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = []
    for err in exc.errors():
        if err["type"] == "json_invalid":
            reason = err.get("ctx", {}).get("error", err["msg"])
            errors.append({"field": "body", "type": "json_invalid", "message": f"Тело запроса не JSON: {reason}"})
            continue
        field = ".".join(str(part) for part in err["loc"][1:]) or err["loc"][0]
        error = {"field": field, "type": err["type"], "message": err["msg"]}
        if err["type"] != "missing":
            error["input"] = input_for_response(err.get("input"))
        errors.append(error)
    # в лог — только поля и типы ошибок, без переданных значений: это могут быть данные клиентов
    logger.warning(
        "%s %s -> 422 validation_error: %s",
        request.method, request.url.path, ", ".join(f"{e['field']} ({e['type']})" for e in errors),
    )
    error = ErrorResponse(code="validation_error", message="Запрос не прошёл проверку", details={"errors": errors})
    return error_response(422, error)


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # Трассировку пишет в лог сам uvicorn, здесь — только к какому запросу она относится
    logger.error("%s %s -> необработанная ошибка %s", request.method, request.url.path, type(exc).__name__)
    error = ErrorResponse(code="internal_error", message="Внутренняя ошибка сервиса, подробности в логе сервера")
    return error_response(500, error)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ChurnServiceError, handle_service_error)
    app.add_exception_handler(HTTPException, handle_http_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
