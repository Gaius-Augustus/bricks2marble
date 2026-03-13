import logging
from pathlib import Path
from timeit import default_timer as clock
from typing import Any


class ElapsedFormatter(logging.Formatter):

    def __init__(self, fmt: str | None = None) -> None:
        super().__init__(fmt)
        self.start_time = clock()

    def format(self, record: logging.LogRecord) -> str:
        timer = getattr(record, "timer", True)
        if timer:
            elapsed = clock() - self.start_time
            record.prefix = f"[{elapsed:0.4f}s] "
        else:
            record.prefix = ""
        return super().format(record)


def setup_logging(
    log_file: str | Path,
    level: int = logging.INFO,
) -> None:
    logger = logging.getLogger()
    logger.setLevel(level)

    handler = logging.FileHandler(log_file)
    formatter = ElapsedFormatter("%(prefix)s%(message)s")
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)


def log_it(message: str, extra: dict[str, Any] | None = None) -> None:
    logging.getLogger().info(message, extra=extra)
