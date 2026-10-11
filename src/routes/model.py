from fastapi import APIRouter

from src.dependencies import ModelServiceDep
from src.models import ModelStatus, TrainMetrics

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train")
def model_train(service: ModelServiceDep) -> TrainMetrics:
    return service.train()


@router.get("/status")
def model_status(service: ModelServiceDep) -> ModelStatus:
    return service.status()
