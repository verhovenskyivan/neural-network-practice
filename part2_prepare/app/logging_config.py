"""Настройка общего формата логирования."""

import logging

from .config import env_level


def configure_logging():
    """Устанавливает формат и уровень диагностических сообщений."""
    level = getattr(logging, env_level(), logging.INFO)
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
