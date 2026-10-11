"""Рекурсивный обход и подготовка числовых CSV или изображений."""

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image


class PreparationError(Exception):
    """Ошибка входных данных или подготовки массива."""


def list_matching_files(directory, extensions):
    """Находит подходящие файлы рекурсивно и сортирует по пути."""
    root = Path(directory)
    if not root.is_dir():
        raise PreparationError(f'Каталог не найден: {root}')
    extensions = {ext.strip().lower().lstrip('.') for ext in extensions.split(',') if ext.strip()}
    if not extensions:
        raise PreparationError('Укажите хотя бы одно расширение.')
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower().lstrip('.') in extensions)
    if not paths:
        raise PreparationError('В каталоге не обнаружены подходящие файлы.')
    return paths


def read_image(filename, size=32):
    """Преобразует изображение в RGB-массив float32 от 0 до 1."""
    with Image.open(filename) as source:
        resized = source.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0


def read_csv(filename):
    """Считывает CSV с заголовком и возвращает двумерный числовой массив."""
    with Path(filename).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise PreparationError(f'В CSV отсутствуют строки данных: {filename}')
    try:
        array = np.asarray([[float(value) for value in row] for row in rows[1:]], dtype=np.float32)
    except ValueError as error:
        raise PreparationError(f'CSV содержит нечисловые значения: {filename}') from error
    if array.ndim != 2 or not np.isfinite(array).all():
        raise PreparationError(f'Необычная форма CSV или NaN: {filename}')
    return array


def prepare_dataset(directory, extensions, output, size=32):
    """Формирует общий .npy и возвращает статистику обработки."""
    start = time.perf_counter()
    paths = list_matching_files(directory, extensions)
    suffixes = {path.suffix.lower() for path in paths}
    is_csv = suffixes == {'.csv'}
    if not is_csv and not suffixes.issubset({'.png', '.jpg', '.jpeg', '.bmp', '.webp'}):
        raise PreparationError('Нельзя смешивать изображения с CSV/другими форматами в одном массиве.')
    if size < 1:
        raise PreparationError('Размер изображения должен быть > 0.')
    samples = []
    try:
        for path in paths:
            sample = read_csv(path) if is_csv else read_image(path, size)
            samples.append(sample)
        if is_csv:
            feature_widths = {array.shape[1] for array in samples}
            if len(feature_widths) != 1:
                raise PreparationError('CSV имеют разное число столбцов.')
            combined = np.concatenate(samples, axis=0)
            minimum = combined.min(axis=0)
            maximum = combined.max(axis=0)
            divisor = np.where(maximum == minimum, 1, maximum - minimum)
            combined = ((combined - minimum) / divisor).astype(np.float32)
        else:
            combined = np.stack(samples).astype(np.float32)
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, combined, allow_pickle=False)
    except (OSError, ValueError, Image.UnidentifiedImageError) as error:
        raise PreparationError(f'Не удалось обработать файлы: {error}') from error
    return {
        'files_count': len(paths),
        'samples_count': int(len(combined)),
        'shape': list(combined.shape),
        'input_bytes': sum(path.stat().st_size for path in paths),
        'output_bytes': destination.stat().st_size,
        'duration_seconds': round(time.perf_counter() - start, 4),
        'output': str(destination),
    }
