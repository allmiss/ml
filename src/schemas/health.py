"""Ответ /health."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelHealth(BaseModel):
    available: bool
    model_type: str | None = None
    trained_at: datetime | None = None


class DatasetHealth(BaseModel):
    loaded: bool
    rows: int | None = None
    # почему датасет недоступен — то же сообщение, что вернули бы /dataset/*
    error: str | None = None


class HealthStatus(BaseModel):
    """Состояние сервиса: ok — есть и модель, и датасет; degraded — чего-то из них нет."""

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "status": "degraded",
            "model": {"available": False, "model_type": None, "trained_at": None},
            "dataset": {"loaded": True, "rows": 2000, "error": None},
        }],
    })

    status: Literal["ok", "degraded"]
    model: ModelHealth
    dataset: DatasetHealth
