"""Logging configuration for the Restaurant Order Demand Prediction system.

Provides structured, consistent console and file logging formatting across
all pipeline modules without relying on unstructured print statements.
"""

import logging
import os
import sys


def configure_logger(
    name: str = "restaurant_demand",
    level: str | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    """Configures and returns a logger instance with standardized formatting.

    Args:
        name: Name of the logger, typically matching the calling module.
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
            Defaults to the LOG_LEVEL environment variable or 'INFO'.
        log_file: Optional path to an output file for persisting logs.

    Returns:
        logging.Logger: Configured logger ready for structured logging.
    """
    resolved_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers if the logger has already been initialized
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
