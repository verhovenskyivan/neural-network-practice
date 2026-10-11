"""Детерминированная CPU-аугментация изображений."""

import numpy as np


def augment_image(index, image, copies=2, seed=123):
    """Возвращает оригинал и copies модифицированных вариантов.

    Применяет повороты на кратные 90 градусов, отражение и шум.
    Результат воспроизводим для одного и того же индекса изображения.
    """
    rng = np.random.default_rng(seed + index)
    result = [np.asarray(image, dtype=np.float32)]
    for _ in range(copies):
        modified = np.rot90(image, k=int(rng.integers(0, 4)))
        if rng.random() > 0.5:
            modified = np.fliplr(modified)
        noise = rng.normal(loc=0, scale=0.035, size=modified.shape)
        result.append(np.clip(modified + noise, 0, 1).astype(np.float32))
    return np.stack(result)


def pool_augment(task):
    """Задача для multiprocessing.Pool: распаковка аргументов аугментации."""
    index, image, copies, seed = task
    return index, augment_image(index, image, copies, seed)
