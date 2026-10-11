from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.exceptions import ChurnServiceError


async def handle_service_error(_request: Request, exc: ChurnServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ChurnServiceError, handle_service_error)
