class ChurnServiceError(Exception):
    """Базовое исключение сервиса.

    Все ошибки сервиса наследуются от него. Обработчик из
    src/handlers/exception_handlers.py превращает их в JSON-ответ
    {"detail": <сообщение>} с HTTP-кодом из status_code.
    """

    status_code = 500


class DatasetUnavailableError(ChurnServiceError):
    """Датасет недоступен.

    Появляется при первом обращении к ChurnDataset.rows, если файл
    датасета не найден или в нём нет ни одной строки данных.
    Эндпоинты /dataset/* и /model/train отвечают 503.
    """

    status_code = 503


class ModelNotTrainedError(ChurnServiceError):
    """Модель ещё не обучена.

    Появляется в ModelService.predict, если в приложении нет модели: при старте
    файл модели не найден или не загрузился, а POST /model/train
    ещё не вызывался. Эндпоинт /predict отвечает 503.
    """

    status_code = 503


class FeatureMismatchError(ChurnServiceError):
    """Признаки для предсказания не совпадают с признаками модели.

    Появляется в predict_churn, если набор или порядок признаков, которые
    сервис собирает из FeatureVectorChurn, отличается от того, на котором
    обучалась загруженная модель. Так бывает, если после обучения изменили
    списки признаков. Эндпоинт /predict отвечает 503, пока модель не переобучат.
    """

    status_code = 503


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


class InvalidModelFileError(ChurnServiceError):
    """Файл модели в неподдерживаемом формате.

    Появляется в load_churn_model, если в файле лежит не TrainedModel,
    например модель, сохранённая старой версией сервиса. При старте
    приложения ошибка пишется в лог, а сервис запускается без модели.
    """
