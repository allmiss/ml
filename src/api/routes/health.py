from fastapi import APIRouter

from src.api.dependencies import DatasetServiceDep, ModelServiceDep
from src.schemas.health import HealthStatus

router = APIRouter(tags=["health"])


@router.get("/health", summary="Состояние сервиса")
def health(model_service: ModelServiceDep, dataset_service: DatasetServiceDep) -> HealthStatus:
    """Доступна ли модель и загружен ли датасет. Всегда отвечает 200, пока сервис
    работает: без модели он жив и может её обучить, поэтому состояние — в поле
    `status`: `ok`, если есть и модель, и датасет, иначе `degraded`."""
    model = model_service.health()
    dataset = dataset_service.health()
    return HealthStatus(
        status="ok" if model.available and dataset.loaded else "degraded",
        model=model,
        dataset=dataset,
    )
