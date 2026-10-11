"""Команды prepare и doctor для анализа и конвертации датасетов."""

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .config import env_path
from .logging_config import configure_logging
from .scanner import PreparationError, prepare_dataset


def doctor():
    """Проверяет Python, пакеты, nvidia-smi/CUDA и рабочую директорию."""
    print('=== Проверка окружения ===')
    print(f'Python: {platform.python_version()}')
    print(f'Платформа: {platform.system()} {platform.machine()}')
    print(f'Рабочая директория: {Path.cwd()}')
    for module in ('numpy', 'PIL', 'tensorflow', 'torch'):
        installed = importlib.util.find_spec(module) is not None
        print(f'Библиотека {module}: {"найдена" if installed else "не установлена"}')
    nvidia_executable = shutil.which('nvidia-smi')
    if nvidia_executable:
        try:
            result = subprocess.run(
                [nvidia_executable, '--query-gpu=name,driver_version', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=6, check=False,
            )
            print('CUDA GPU (nvidia-smi): ' + (result.stdout.strip() if result.returncode == 0 else 'не определён'))
        except (OSError, subprocess.TimeoutExpired):
            print('CUDA GPU: проверка завершилась ошибкой.')
    else:
        print('CUDA GPU: nvidia-smi отсутствует (для NumPy CPU не требуется).')


def main(argv=None):
    """Разбирает argparse и запускает prepare либо doctor."""
    configure_logging()
    parser = argparse.ArgumentParser(description='Подготовка данных для нейросети')
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare = subparsers.add_parser('prepare', help='Создать нормализованный массив NumPy')
    prepare.add_argument('--path', default=str(env_path('NN_INPUT_PATH', 'data/input/images')))
    prepare.add_argument('--ext', default=os.getenv('NN_EXTENSIONS', '.png,.jpg,.jpeg'))
    prepare.add_argument('--output', default=str(env_path('NN_OUTPUT_PATH', 'data/output/prepared.npy')))
    prepare.add_argument('--size', type=int, default=int(os.getenv('NN_IMAGE_SIZE', '32')))
    subparsers.add_parser('doctor', help='Проверить окружение и инструменты')
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        doctor()
        return 0
    try:
        stats = prepare_dataset(args.path, args.ext, args.output, args.size)
        print('=== Данные подготовлены ===')
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0
    except PreparationError as error:
        print(f'Ошибка подготовки: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
