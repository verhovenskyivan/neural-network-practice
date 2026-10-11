"""Загрузка и нормализация CSV без утечки данных из test в train."""

import csv
from pathlib import Path

import numpy as np

from .exceptions import DataFileError, MismatchedDataError


class DatasetManager:
    """Управляет CSV-данными и статистиками стандартизации."""

    def __init__(self):
        self.features = None
        self.labels = None
        self.scaler = None

    def load_csv(self, path):
        """Читает числовые CSV, где последний столбец содержит 0 или 1."""
        try:
            with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
                table = list(csv.reader(stream))
            if len(table) < 3:
                raise MismatchedDataError('CSV должен содержать заголовок и минимум две записи.')
            data = np.asarray([[float(value) for value in row] for row in table[1:]], dtype=float)
            if data.ndim != 2 or data.shape[1] < 2 or not np.isfinite(data).all():
                raise MismatchedDataError('В CSV ожидаются числовые признаки и последний столбец-метка.')
            self.features = data[:, :-1]
            self.labels = data[:, -1].reshape(-1, 1)
            if not np.isin(self.labels, [0, 1]).all():
                raise MismatchedDataError('Метки бинарной классификации должны быть 0 или 1.')
            return len(data), self.features.shape[1]
        except (OSError, UnicodeError, ValueError, csv.Error) as error:
            raise DataFileError(f'Ошибка чтения CSV: {error}') from error

    def split(self, test_ratio=0.25, seed=42):
        """Перемешивает, разделяет данные и стандартизует по статистике train."""
        if self.features is None:
            raise MismatchedDataError('Сначала загрузите CSV.')
        if not 0 < test_ratio < 1:
            raise ValueError('Доля test должна находиться между 0 и 1.')
        order = np.random.default_rng(seed).permutation(len(self.features))
        size_test = max(1, int(round(len(order) * test_ratio)))
        if size_test >= len(order):
            raise MismatchedDataError('Слишком мало наблюдений для разбиения.')
        test_indices = order[:size_test]
        train_indices = order[size_test:]
        train = self.features[train_indices]
        mean = train.mean(axis=0)
        deviation = train.std(axis=0)
        deviation[deviation == 0] = 1
        self.scaler = {'mean': mean.tolist(), 'std': deviation.tolist()}
        return self.transform(train), self.labels[train_indices], self.transform(self.features[test_indices]), self.labels[test_indices]

    def transform(self, features):
        """Стандартизует признаки по параметрам обучающей выборки."""
        if self.scaler is None:
            raise MismatchedDataError('Параметры нормализации отсутствуют. Обучите сеть или загрузите веса.')
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != len(self.scaler['mean']):
            raise MismatchedDataError('Число входных признаков не совпадает с параметрами нормализации.')
        return (features - np.array(self.scaler['mean'])) / np.array(self.scaler['std'])
