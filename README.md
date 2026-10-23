# ML Churn Service

Сервис на FastAPI, который предсказывает отток клиентов (churn): по данным клиента — тариф, активность, платежи — оценивает вероятность того, что он уйдёт. Сервис обучает модель на `churn_dataset.csv`, хранит историю обучений с метриками и отдаёт предсказания по HTTP.

Учебный проект; ход работы по дням — в [docs/homework.md](docs/homework.md).

## Эндпоинты

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/health` | состояние сервиса: доступна ли модель, загружен ли датасет |
| GET | `/dataset/preview?n=5` | первые строки датасета |
| GET | `/dataset/info` | размер датасета и распределение `churn` |
| GET | `/dataset/split-info` | размеры обучающей и тестовой выборок |
| POST | `/model/train` | обучить модель, вернуть метрики на тестовой выборке |
| GET | `/model/status` | текущая модель: тип, гиперпараметры, время обучения, метрики |
| GET | `/model/schema` | признаки, их типы и допустимые значения |
| GET | `/model/metrics?last=5` | метрики последнего обучения и история |
| GET | `/model/compare?metric=f1` | настройки модели из истории, от лучшей к худшей |
| POST | `/predict` | предсказание для одного клиента или списка |

Интерактивная документация с примерами запросов и ошибок — `/docs`.

## Датасет `churn_dataset.csv`

Файл `src/data/churn_dataset.csv`: 2000 клиентов, 20.1% из них ушли. Одна строка — один клиент:

| Колонка | Тип | Значения | Что значит |
|---|---|---|---|
| `monthly_fee` | число | 9.99, 19.99, 29.99, 49.99 | ежемесячная плата по тарифу |
| `usage_hours` | число | 0 – 52 | часы использования сервиса |
| `support_requests` | целое | 0 – 4 | обращения в поддержку |
| `account_age_months` | целое | 1 – 39 | сколько месяцев клиент с сервисом |
| `failed_payments` | целое | 0 – 2 | неуспешные платежи |
| `region` | строка | `africa`, `america`, `asia`, `europe` | регион |
| `device_type` | строка | `desktop`, `mobile`, `tablet` | устройство |
| `payment_method` | строка | `card`, `crypto`, `paypal` | способ оплаты |
| `autopay_enabled` | 0 / 1 | | включён ли автоплатёж |
| `churn` | 0 / 1 | | целевая переменная: 1 — клиент ушёл |

```csv
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
9.99,27.92,1,14,1,america,desktop,card,1,1
19.99,21.48,2,1,0,america,mobile,card,1,0
```

Строки без `churn` при обучении отбрасываются, пропуски в признаках заполняются медианой или самым частым значением. Для `/predict` нужны те же признаки без `churn`.

## Запуск локально

Нужен Python 3.14.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload
```

Сервис запустится на http://127.0.0.1:8000, документация — http://127.0.0.1:8000/docs. Модель и история обучений сохраняются в папку `models/` и подхватываются при следующем запуске.

Тесты:

```bash
pip install pytest httpx2
python -m pytest
```

## Запуск в Docker

```bash
docker build -t churn-service .
docker run -p 8000:8000 -v churn-models:/app/models churn-service
```

Volume `churn-models` сохраняет модель и историю обучений между перезапусками контейнера; без него они пропадут вместе с контейнером. Docker сам проверяет `/health` внутри контейнера, статус виден в `docker ps`. Уровень логов задаётся переменной окружения `LOG_LEVEL` (по умолчанию `INFO`): `docker run -e LOG_LEVEL=DEBUG ...`.

## Примеры запросов

### Обучение — `POST /model/train`

Без тела обучается логистическая регрессия с параметрами по умолчанию:

```bash
curl -X POST http://127.0.0.1:8000/model/train
```
```json
{"accuracy": 0.787, "f1": 0.045, "roc_auc": 0.609}
```

Тип модели (`logreg` или `random_forest`) и гиперпараметры sklearn можно передать в теле. Уходят 20% клиентов, поэтому без `class_weight` модель почти всегда отвечает «останется» — у неё высокая accuracy, но f1 близок к нулю:

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logreg", "hyperparameters": {"class_weight": "balanced"}}'
```
```json
{"accuracy": 0.59, "f1": 0.354, "roc_auc": 0.611}
```

Каждое обучение записывается в историю: `GET /model/metrics?last=5` — последние записи, `GET /model/compare` — сравнение настроек.

### Предсказание — `POST /predict`

Один клиент:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_fee": 29.99, "usage_hours": 3.5, "support_requests": 4, "account_age_months": 2,
       "failed_payments": 2, "region": "asia", "device_type": "mobile", "payment_method": "crypto",
       "autopay_enabled": 0}'
```
```json
{"predictions": [{"churn": 1, "probabilities": {"0": 0.1145, "1": 0.8855}}]}
```

Список клиентов — ответ в том же порядке:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"monthly_fee": 29.99, "usage_hours": 3.5, "support_requests": 4, "account_age_months": 2,
        "failed_payments": 2, "region": "asia", "device_type": "mobile", "payment_method": "crypto",
        "autopay_enabled": 0},
       {"monthly_fee": 29.99, "usage_hours": 40.0, "support_requests": 0, "account_age_months": 36,
        "failed_payments": 0, "region": "asia", "device_type": "mobile", "payment_method": "card",
        "autopay_enabled": 1}]'
```
```json
{"predictions": [
  {"churn": 1, "probabilities": {"0": 0.1145, "1": 0.8855}},
  {"churn": 0, "probabilities": {"0": 0.9225, "1": 0.0775}}
]}
```

Ответы выше — для модели `logreg` с `class_weight=balanced`.

### Ошибки

Все ошибки приходят в одном формате: `code` — машинный код, `message` — текст для человека, `details` — подробности.

```json
{"code": "model_not_trained", "message": "Модель не обучена, обучите её через POST /model/train", "details": null}
```
```json
{"code": "invalid_training_config", "message": "Неизвестный тип модели 'xgboost'",
 "details": {"model_type": "xgboost", "supported": ["logreg", "random_forest"]}}
```
```json
{"code": "validation_error", "message": "Запрос не прошёл проверку",
 "details": {"errors": [{"field": "0.monthly_fee", "type": "float_parsing",
                         "message": "Input should be a valid number, unable to parse string as a number", "input": "abc"}]}}
```

Все коды ошибок с примерами — в `/docs`.

## Структура проекта

```
src/
  main.py          сборка приложения
  api/             роуты, зависимости, обработчики ошибок
  services/        связка API и ML: обучение, предсказание, история
  ml/              churn pipeline: загрузка датасета, подготовка, обучение, предсказание, сохранение модели
  schemas/         pydantic-схемы запросов и ответов
  core/            настройки, исключения, логирование
  data/            churn_dataset.csv
tests/
  unit/            тесты функций без FastAPI
  api/             тесты эндпоинтов через TestClient
docs/homework.md   ход работы по дням
Dockerfile
requirements.txt
```
