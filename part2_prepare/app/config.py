"""Безопасные настройки CLI: аргументы, ENV, значения по умолчанию."""

import os
from pathlib import Path


def env_path(name, fallback):
    """Возвращает путь из переменной среды или путь по умолчанию."""
    return Path(os.getenv(name, fallback))


def env_level():
    """Возвращает настроенный уровень логирования."""
    return os.getenv('NN_LOG_LEVEL', 'INFO').upper()
