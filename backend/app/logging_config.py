import logging
import sys

from app.config import settings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def job_logger(job_id: str) -> logging.Logger:
    """Per-job logger that writes to storage/logs/<job_id>.log in addition to stdout."""
    logger = logging.getLogger(f"job.{job_id}")
    if not logger.handlers:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = settings.logs_dir / f"{job_id}.log"
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = True
    return logger
