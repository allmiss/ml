from typing import Annotated

from fastapi import Depends, Request

from src.services.dataset_service import DatasetService
from src.services.model_service import ModelService


def get_dataset_service(request: Request) -> DatasetService:
    return request.app.state.dataset_service


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]
ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]
