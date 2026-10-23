from fastapi import APIRouter, Query

from src.api.dependencies import DatasetServiceDep
from src.schemas.churn import DatasetRowChurn
from src.schemas.dataset import DatasetInfo, SplitInfo

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/preview")
def dataset_preview(service: DatasetServiceDep, n: int = Query(5, ge=1)) -> list[DatasetRowChurn]:
    return service.preview(n)


@router.get("/info")
def dataset_info(service: DatasetServiceDep) -> DatasetInfo:
    return service.info()


@router.get("/split-info")
def dataset_split_info(service: DatasetServiceDep) -> SplitInfo:
    return service.split_info()
