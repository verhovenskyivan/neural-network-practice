"""Полносвязная сеть для бинарной классификации на NumPy."""

import json
from pathlib import Path

import numpy as np

from .exceptions import DataFileError, InvalidLayerSizeError, MismatchedDataError


class NeuralNetwork:
    """MLP: входной слой, один скрытый слой и один сигмоидный выход.

    Args:
        input_size: Число входных признаков.
        hidden_size: Число скрытых нейронов.
        activation: Функция активации скрытого слоя: relu или sigmoid.
        seed: Зерно случайного генератора для воспроизводимости.
    """

    def __init__(self, input_size, hidden_size, activation='relu', seed=42):
        if input_size < 1 or hidden_size < 1:
            raise InvalidLayerSizeError('Количество нейронов должно быть положительным.')
        if activation not in ('relu', 'sigmoid'):
            raise NeuralNetworkErrorActivation('Доступны только relu и sigmoid.')
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.activation = activation
        random_generator = np.random.default_rng(seed)
        factor = np.sqrt(2 / input_size) if activation == 'relu' else np.sqrt(1 / input_size)
        self.weight_hidden = random_generator.normal(0, factor, (input_size, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weight_output = random_generator.normal(0, np.sqrt(1 / hidden_size), (hidden_size, 1))
        self.bias_output = np.zeros((1, 1))
        self.history = []

    @staticmethod
    def _sigmoid(values):
        """Возвращает сигмоиду с защитой экспоненты от переполнения."""
        return 1 / (1 + np.exp(-np.clip(values, -500, 500)))

    def _activate(self, values):
        """Применяет выбранную функцию активации скрытого слоя."""
        return np.maximum(0, values) if self.activation == 'relu' else self._sigmoid(values)

    def _activation_gradient(self, raw_values, activated_values):
        """Находит производную функции активации скрытого слоя."""
        return (raw_values > 0).astype(float) if self.activation == 'relu' else activated_values * (1 - activated_values)

    def forward(self, features):
        """Возвращает вероятности класса 1 для матрицы входных признаков."""
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise MismatchedDataError(f'Ожидается матрица (N, {self.input_size}); получено {features.shape}.')
        raw_hidden = features @ self.weight_hidden + self.bias_hidden
        hidden = self._activate(raw_hidden)
        output = self._sigmoid(hidden @ self.weight_output + self.bias_output)
        return output, (features, raw_hidden, hidden)

    @staticmethod
    def _binary_cross_entropy(prediction, labels):
        """Возвращает среднюю бинарную кросс-энтропию."""
        clipped = np.clip(prediction, 1e-8, 1 - 1e-8)
        return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))

    def train(self, features, labels, epochs=100, learning_rate=0.1, batch_size=16, verbose=True):
        """Обучает сеть SGD с мини-батчами и сохраняет историю ошибки."""
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=float).reshape(-1, 1)
        if features.ndim != 2 or features.shape[0] != labels.shape[0] or features.shape[1] != self.input_size:
            raise MismatchedDataError('Размерности признаков и целевой переменной не совпадают.')
        if not np.isin(labels, [0, 1]).all():
            raise MismatchedDataError('Для бинарной классификации допустимы только метки 0 и 1.')
        if epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError('Эпохи и батч должны быть > 0, скорость обучения > 0.')
        rng = np.random.default_rng(17)
        self.history = []
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(features))
            for start in range(0, len(features), batch_size):
                indices = order[start:start + batch_size]
                batch_features = features[indices]
                batch_labels = labels[indices]
                predictions, (_, raw_hidden, hidden) = self.forward(batch_features)
                count = len(indices)
                grad_output = (predictions - batch_labels) / count
                dw_output = hidden.T @ grad_output
                db_output = grad_output.sum(axis=0, keepdims=True)
                grad_hidden = (grad_output @ self.weight_output.T) * self._activation_gradient(raw_hidden, hidden)
                dw_hidden = batch_features.T @ grad_hidden
                db_hidden = grad_hidden.sum(axis=0, keepdims=True)
                self.weight_output -= learning_rate * dw_output
                self.bias_output -= learning_rate * db_output
                self.weight_hidden -= learning_rate * dw_hidden
                self.bias_hidden -= learning_rate * db_hidden
            full_prediction, _ = self.forward(features)
            loss = self._binary_cross_entropy(full_prediction, labels)
            self.history.append(loss)
            if verbose and (epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs):
                print(f'Эпоха {epoch:>4}/{epochs}: BCE={loss:.5f}')
        return self.history

    def predict(self, features):
        """Возвращает предсказанные вероятности и метки по порогу 0.5."""
        probabilities, _ = self.forward(features)
        return probabilities.ravel(), (probabilities.ravel() >= 0.5).astype(int)

    def save(self, filename, scaler=None):
        """Сохраняет веса, архитектуру и необязательные параметры нормализации в JSON."""
        payload = {
            'input_size': self.input_size, 'hidden_size': self.hidden_size,
            'activation': self.activation,
            'weight_hidden': self.weight_hidden.tolist(), 'bias_hidden': self.bias_hidden.tolist(),
            'weight_output': self.weight_output.tolist(), 'bias_output': self.bias_output.tolist(),
            'history': self.history, 'scaler': scaler,
        }
        target = Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, filename):
        """Восстанавливает обученную модель и параметры нормализации из JSON."""
        try:
            payload = json.loads(Path(filename).read_text(encoding='utf-8'))
            model = cls(payload['input_size'], payload['hidden_size'], payload['activation'])
            for name in ('weight_hidden', 'bias_hidden', 'weight_output', 'bias_output'):
                setattr(model, name, np.asarray(payload[name], dtype=float))
            if model.weight_hidden.shape != (model.input_size, model.hidden_size):
                raise MismatchedDataError('Повреждён файл: некорректная форма матрицы весов.')
            if model.weight_output.shape != (model.hidden_size, 1):
                raise MismatchedDataError('Повреждён файл: некорректная форма выходных весов.')
            model.history = payload.get('history', [])
            return model, payload.get('scaler')
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise DataFileError(f'Не удалось загрузить веса: {error}') from error


class NeuralNetworkErrorActivation(InvalidLayerSizeError):
    """Ошибка названия функции активации."""
