import logging
import sys

from src.config import LOG_LEVEL

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = LOG_LEVEL) -> None:
    """Логи сервиса — все логгеры src.* — в stdout в одном формате.

    uvicorn настраивает только свои логгеры, поэтому без этой настройки
    сообщения сервиса уровня INFO никуда бы не выводились. Повторный вызов
    (например, при перезагрузке с --reload) не добавляет второй обработчик.
    """
    logger = logging.getLogger("src")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
