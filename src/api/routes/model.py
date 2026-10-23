from typing import Annotated

from fastapi import APIRouter, Query

from src.api.dependencies import ModelServiceDep
from src.api.routes.error_examples import TRAIN_ERRORS
from src.ml.training import ModelType
from src.schemas.model import (
    MetricName,
    ModelComparison,
    ModelMetrics,
    ModelSchema,
    ModelStatus,
    TrainingConfigChurn,
    TrainMetrics,
)

router = APIRouter(prefix="/model", tags=["model"])


@router.post(
    "/train",
    summary="Обучить модель",
    responses=TRAIN_ERRORS,
)
def model_train(service: ModelServiceDep, config: TrainingConfigChurn | None = None) -> TrainMetrics:
    """Обучает модель выбранного типа на churn_dataset.csv, сохраняет её на диск
    и возвращает метрики на тестовой выборке. Без тела запроса обучается logreg
    с параметрами по умолчанию."""
    return service.train(config or TrainingConfigChurn())


@router.get("/status")
def model_status(service: ModelServiceDep) -> ModelStatus:
    return service.status()


@router.get("/metrics", summary="Метрики последнего обучения и история")
def model_metrics(
    service: ModelServiceDep,
    last: Annotated[int, Query(ge=0, le=100, description="Сколько последних записей истории вернуть")] = 0,
    model_type: Annotated[ModelType | None, Query(description="Только обучения этого типа модели")] = None,
) -> ModelMetrics:
    """`latest` — запись о последнем успешном обучении: время, тип модели,
    гиперпараметры, размеры выборок и метрики accuracy, f1, roc_auc.
    Пока модель ни разу не обучалась, там `null`.
    `history` — `last` последних записей, новые первыми; по умолчанию пустой список.
    С `model_type` и `latest`, и `history` берутся только из обучений этого типа модели."""
    return service.metrics(last, model_type)


@router.get("/compare", summary="Сравнить настройки модели по истории обучений")
def model_compare(
    service: ModelServiceDep,
    metric: Annotated[MetricName, Query(description="Метрика, по которой сравниваются настройки")] = "f1",
    model_type: Annotated[ModelType | None, Query(description="Сравнивать только настройки этого типа модели")] = None,
) -> ModelComparison:
    """Сводит историю обучений по настройкам (тип модели + гиперпараметры) и
    сортирует их по `metric`, лучшие первыми; `best` — лучшая настройка.
    Повторные обучения с той же настройкой не дублируются: `runs` — сколько их было,
    метрики — по последнему. По умолчанию сравнение по f1: при 20% ушедших клиентов
    accuracy выше у модели, которая почти всегда отвечает «останется»."""
    return service.compare(metric, model_type)


@router.get("/schema", summary="Схема признаков для /predict")
def model_schema(service: ModelServiceDep) -> ModelSchema:
    """Список признаков в том порядке, в котором их использует модель, с JSON-типами.
    Для категориальных признаков `allowed_values` — значения, известные обученной
    модели; пока модель не обучена, там `null`. Неизвестные значения не вызывают
    ошибку, но модель их не учитывает."""
    return service.schema()
