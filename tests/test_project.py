"""Базовые регрессионные проверки всех трёх частей практикума."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from part1_mlp.app.exceptions import MismatchedDataError
from part1_mlp.app.manager import DatasetManager
from part1_mlp.app.models import NeuralNetwork
from part2_prepare.app.scanner import prepare_dataset
from part3_parallel.app.augment import augment_image


class ProjectTests(unittest.TestCase):
    """Проверяет функции сети, подготовки файлов и аугментации."""

    def test_forward_and_shape_error(self):
        network = NeuralNetwork(2, 8)
        probability, _ = network.forward(np.zeros((3, 2)))
        self.assertEqual(probability.shape, (3, 1))
        with self.assertRaises(MismatchedDataError):
            network.forward(np.zeros((3, 4)))

    def test_learning_save_and_load(self):
        features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        labels = np.array([0, 0, 0, 1])
        network = NeuralNetwork(2, 8)
        history = network.train(features, labels, epochs=140, learning_rate=.3, batch_size=4, verbose=False)
        self.assertLess(history[-1], history[0])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'weights.json'
            network.save(output, {'mean': [0, 0], 'std': [1, 1]})
            loaded, scaler = NeuralNetwork.load(output)
            self.assertEqual(scaler['mean'], [0, 0])
            np.testing.assert_allclose(network.predict(features)[0], loaded.predict(features)[0])

    def test_prepare_images(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'nested').mkdir()
            Image.new('RGB', (10, 10), (255, 0, 0)).save(root / 'nested' / 'first.png')
            Image.new('RGB', (9, 14), (0, 128, 0)).save(root / 'second.png')
            stats = prepare_dataset(root, '.png', root / 'result.npy', size=16)
            self.assertEqual(stats['shape'], [2, 16, 16, 3])
            array = np.load(root / 'result.npy')
            self.assertTrue(0 <= array.min() <= array.max() <= 1)

    def test_augmentation_is_deterministic(self):
        image = np.ones((12, 12, 3), dtype=np.float32)
        first = augment_image(3, image, copies=2)
        second = augment_image(3, image, copies=2)
        self.assertEqual(first.shape, (3, 12, 12, 3))
        np.testing.assert_array_equal(first, second)


if __name__ == '__main__':
    unittest.main()
