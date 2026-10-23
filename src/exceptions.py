from typing import Any


class ChurnServiceError(Exception):
    """Базовое исключение сервиса.

    Все ошибки сервиса наследуются от него. Обработчик из
    src/handlers/exception_handlers.py превращает их в JSON-ответ
    ErrorResponse {"code", "message", "details"} с HTTP-кодом из status_code.
    code — постоянный машинный код ошибки, message — текст для человека,
    details — необязательные подробности (что именно пошло не так).
    """

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details


class DatasetUnavailableError(ChurnServiceError):
    """Датасет недоступен.

    Появляется при первом обращении к ChurnDataset.rows, если файл
    датасета не найден или в нём нет ни одной строки данных.
    Эндпоинты /dataset/* и /model/train отвечают 503.
    """

    status_code = 503
    code = "dataset_unavailable"


class ModelNotTrainedError(ChurnServiceError):
    """Модель ещё не обучена.

    Появляется в ModelService.predict, если в приложении нет модели: при старте
    файл модели не найден или не загрузился, а POST /model/train
    ещё не вызывался. Эндпоинт /predict отвечает 503.
    """

    status_code = 503
    code = "model_not_trained"


class FeatureMismatchError(ChurnServiceError):
    """Признаки для предсказания не совпадают с признаками модели.

    Появляется в predict_churn, если набор или порядок признаков, которые
    сервис собирает из FeatureVectorChurn, отличается от того, на котором
    обучалась загруженная модель. Так бывает, если после обучения изменили
    списки признаков. Эндпоинт /predict отвечает 503, пока модель не переобучат.
    """

    status_code = 503
    code = "feature_mismatch"


class InvalidTrainingConfigError(ChurnServiceError):
    """Конфигурация обучения не подходит.

    Появляется в resolve_config, если model_type не поддерживается, в
    create_model, если в hyperparameters есть параметр, которого нет у
    выбранной модели, и в
    train_churn_model, если модель отвергла значение гиперпараметра при
    обучении (например, C=-1). Эндпоинт /model/train отвечает 400,
    сохранённая модель при этом не меняется.
    """

    status_code = 400
    code = "invalid_training_config"


class InvalidModelFileError(ChurnServiceError):
    """Файл модели в неподдерживаемом формате.

    Появляется в load_churn_model, если в файле лежит не TrainedModel,
    например модель, сохранённая старой версией сервиса. При старте
    приложения ошибка пишется в лог, а сервис запускается без модели.
    """

    code = "invalid_model_file"


class DataPreparationError(ChurnServiceError):
    """Не удалось подготовить данные для обучения.

    Появляется в ChurnDataset, если строка CSV не проходит проверку
    DatasetRowChurn (например, текст в числовой колонке), в prepare_data,
    если в данных нет нужных колонок, и в split_data, если данные нельзя
    разбить на выборки (слишком мало строк или клиентов одного из классов).
    Эндпоинты /dataset/* и /model/train отвечают 500, сохранённая модель
    при этом не меняется.
    """

    status_code = 500
    code = "data_preparation_failed"


class PredictionError(ChurnServiceError):
    """Модель не смогла посчитать предсказание.

    Появляется в predict_churn, если pipeline.predict или predict_proba
    упали на переданных данных. Эндпоинт /predict отвечает 500.
    """

    status_code = 500
    code = "prediction_failed"


class InvalidHistoryFileError(ChurnServiceError):
    """Файл истории обучений повреждён.

    Появляется в TrainingHistory при первом чтении истории, если файл
    не является JSON-списком записей TrainingRecord (например, его
    отредактировали вручную). Файл при этом не перезаписывается, чтобы
    не потерять историю; его нужно исправить или удалить.
    """

    status_code = 500
    code = "invalid_history_file"
