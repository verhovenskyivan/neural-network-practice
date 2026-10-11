"""Текстовое меню для работы с бинарным MLP."""

from pathlib import Path

import numpy as np

from .exceptions import NeuralNetworkError
from .manager import DatasetManager
from .models import NeuralNetwork
from .utils import save_loss_plot


def _ask_integer(question):
    """Читает положительное целое число из консоли."""
    number = int(input(question).strip())
    if number < 1:
        raise ValueError('Ожидается положительное целое число.')
    return number


def main():
    """Запускает меню и обрабатывает ошибки без аварийного выхода."""
    manager = DatasetManager()
    network = None
    train_features = train_labels = test_features = test_labels = None
    while True:
        print('\n=== MLP: личная лаборатория ===')
        print('1. Создать сеть')
        print('2. Загрузить CSV и разбить train/test')
        print('3. Обучить сеть')
        print('4. Предсказание (вручную или из CSV)')
        print('5. Сохранить/загрузить веса')
        print('6. Показать историю и сохранить график ошибки')
        print('7. Выход')
        try:
            command = input('Выберите пункт: ').strip()
            if command == '1':
                input_size = _ask_integer('Входных признаков: ')
                hidden_size = _ask_integer('Нейронов скрытого слоя: ')
                activation = input('Активация (relu/sigmoid): ').strip().lower()
                network = NeuralNetwork(input_size, hidden_size, activation)
                print(f'Сеть создана: {input_size} -> {hidden_size} -> 1, {activation}.')
            elif command == '2':
                filename = input('Путь к CSV: ').strip()
                count, features_count = manager.load_csv(filename)
                train_features, train_labels, test_features, test_labels = manager.split()
                print(f'Загружено {count} строк, признаков: {features_count}.')
                print(f'Обучающая выборка: {len(train_features)}, тестовая: {len(test_features)}.')
                print('Стандартизация рассчитана только по train.')
            elif command == '3':
                if network is None or train_features is None:
                    print('Сначала создайте сеть и загрузите CSV.')
                    continue
                lr = float(input('Скорость обучения: '))
                epochs = _ask_integer('Количество эпох: ')
                batch_size = _ask_integer('Размер батча: ')
                network.train(train_features, train_labels, epochs, lr, batch_size)
                _, labels = network.predict(test_features)
                accuracy = (labels == test_labels.ravel()).mean()
                print(f'Точность на тесте: {accuracy:.3f} ({len(test_labels)} примеров).')
            elif command == '4':
                if network is None or manager.scaler is None:
                    print('Необходимо обучить/загрузить сеть с параметрами нормализации.')
                    continue
                source = input('Источник: 1 - вручную, 2 - CSV: ').strip()
                if source == '1':
                    values = [float(value) for value in input('Признаки через запятую: ').split(',')]
                    features = np.array([values], dtype=float)
                elif source == '2':
                    filename = input('Путь к CSV без заголовка (только признаки): ').strip()
                    features = np.loadtxt(filename, delimiter=',', ndmin=2)
                else:
                    raise ValueError('Выберите 1 или 2.')
                probabilities, labels = network.predict(manager.transform(features))
                for probability, label in zip(probabilities, labels):
                    print(f'Вероятность 1 = {probability:.4f}, предсказанный класс = {label}')
            elif command == '5':
                mode = input('1 - сохранить, 2 - загрузить: ').strip()
                path = input('Путь к JSON: ').strip()
                if mode == '1':
                    if network is None:
                        print('Сначала создайте сеть.')
                        continue
                    network.save(path, manager.scaler)
                    print(f'Модель сохранена: {Path(path)}')
                elif mode == '2':
                    network, manager.scaler = NeuralNetwork.load(path)
                    print(f'Модель загружена: {network.input_size} -> {network.hidden_size} -> 1.')
                else:
                    raise ValueError('Выберите 1 или 2.')
            elif command == '6':
                if network is None or not network.history:
                    print('История пока отсутствует.')
                    continue
                print(f'Ошибка: начало={network.history[0]:.5f}, конец={network.history[-1]:.5f}.')
                filename = input('Путь к PNG: ').strip()
                save_loss_plot(network.history, filename)
                print(f'График сохранён: {filename}')
            elif command == '7':
                print('Выход из программы.')
                break
            else:
                print('Неизвестный пункт меню.')
        except (NeuralNetworkError, OSError, ValueError, RuntimeError) as error:
            print(f'Ошибка: {error}')
        except EOFError:
            print('\nЗавершение ввода.')
            break


if __name__ == '__main__':
    main()
