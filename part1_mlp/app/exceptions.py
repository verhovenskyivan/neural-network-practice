"""Понятные исключения для консольной нейросети."""


class NeuralNetworkError(Exception):
    """Базовая ошибка нейросетевого приложения."""


class InvalidLayerSizeError(NeuralNetworkError):
    """Размер слоя нейросети некорректен."""


class MismatchedDataError(NeuralNetworkError):
    """Форма или содержимое входных данных не соответствует модели."""


class DataFileError(NeuralNetworkError):
    """Ошибка чтения набора данных или сохранённой модели."""
