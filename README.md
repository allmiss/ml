# ML Churn Service

Сервис для предсказания оттока клиентов на FastAPI.

## Задачи

### День 1. Каркас сервиса

1. Установить fastapi и uvicorn — сделано
2. Создать файл main.py — сделано (`src/main.py`)
3. Создать объект приложения `app = FastAPI()` — сделано
4. Добавить эндпоинт `GET /`, который возвращает `{"message": "ml churn service is running"}` — сделано

### День 2. Модели данных

1. Создать Pydantic-модель `FeatureVectorChurn` — сделано (`src/models.py`)
   - monthly_fee float
   - usage_hours float
   - support_requests int
   - account_age_months int
   - failed_payments int
   - region str
   - device_type str
   - payment_method str
   - autopay_enabled int
2. Создать модель `DatasetRowChurn` для строки тренировочного датасета: те же признаки плюс поле `churn int` — сделано
3. Добавить временный эндпоинт `POST /predict`, который принимает `FeatureVectorChurn` и возвращает те же данные — сделано
4. Проверить структуру входных и выходных данных через `/docs` — сделано: вход и выход `/predict` — схема `FeatureVectorChurn`, все 9 полей обязательные, при неверных данных возвращается 422

### День 3. Датасет

1. Положить файл churn_dataset.csv в папку проекта — сделано (`src/data/churn_dataset.csv`)
2. Добавить модуль для работы с датасетом — сделано (`src/dataset.py`)
   - читать CSV файл
   - преобразовывать строки в объекты `DatasetRowChurn`
   - хранить данные в списке словарей
3. Добавить эндпоинт `GET /dataset/preview`, который возвращает первые N строк из churn_dataset.csv в виде JSON — сделано (`?n=`, по умолчанию 5)
4. Добавить эндпоинт `GET /dataset/info`: количество строк, количество столбцов, список признаков, распределение `churn` по классам — сделано
5. Убедиться, что загрузка и просмотр датасета не приводят к ошибкам — сделано
