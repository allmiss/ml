"""Общий формат ошибки сервиса."""

from typing import Any

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Общий формат ошибки: машинный код, сообщение для человека и подробности."""

    code: str
    message: str
    details: dict[str, Any] | None = None
