# Часть 2 — CLI подготовки данных

Команды из корня: `python -m part2_prepare.app.cli doctor` и
`python -m part2_prepare.app.cli prepare --path data/input/images --ext .png --output data/output/prepared.npy --size 32`.
Рекурсивный обход pathlib, преобразование изображений к RGB 32×32, нормализация [0,1], статистика размеров и времени.
CSV с числовыми столбцами также поддерживается, но не следует смешивать изображения и таблицы в одной партии.
Переменные окружения описаны в `.env.example`.
