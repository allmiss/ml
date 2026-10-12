from fastapi import APIRouter

from src.dependencies import ModelServiceDep
from src.models import ErrorResponse, ModelSchema, ModelStatus, TrainingConfigChurn, TrainMetrics

router = APIRouter(prefix="/model", tags=["model"])


@router.post(
    "/train",
    summary="Обучить модель",
    responses={400: {"model": ErrorResponse, "description": "Неизвестный тип модели или неверные гиперпараметры"}},
)
def model_train(service: ModelServiceDep, config: TrainingConfigChurn | None = None) -> TrainMetrics:
    """Обучает модель выбранного типа на churn_dataset.csv, сохраняет её на диск
    и возвращает метрики на тестовой выборке. Без тела запроса обучается logreg
    с параметрами по умолчанию."""
    return service.train(config or TrainingConfigChurn())


@router.get("/status")
def model_status(service: ModelServiceDep) -> ModelStatus:
    return service.status()


@router.get("/schema", summary="Схема признаков для /predict")
def model_schema(service: ModelServiceDep) -> ModelSchema:
    """Список признаков в том порядке, в котором их использует модель, с JSON-типами.
    Для категориальных признаков `allowed_values` — значения, известные обученной
    модели; пока модель не обучена, там `null`. Неизвестные значения не вызывают
    ошибку, но модель их не учитывает."""
    return service.schema()
