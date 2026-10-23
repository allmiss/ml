FROM python:3.14-slim

# без .pyc в образе и с выводом логов сразу, без буфера
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# зависимости — отдельным слоем до кода: при правках кода они не переустанавливаются
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# код сервиса вместе с датасетом src/data/churn_dataset.csv
COPY src ./src

# сервис работает не от root; models/ — куда он сохраняет модель и историю обучений
RUN useradd --create-home churn && mkdir models && chown churn models
USER churn

EXPOSE 8000

# Docker сам опрашивает /health изнутри контейнера; /health отвечает 200, пока сервис
# работает, даже без модели, поэтому unhealthy — только если сервис не отвечает
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
