# Часть 3 — потоки и процессы

Запуск: `python -m part3_parallel.app.performance --path data/input/images --threads 3 --processes 2 --copies 2`.
Поток producer заполняет `queue.Queue`, потоки consumer читают PIL, `Lock` защищает счётчик,
`multiprocessing.Pool.apply_async` распределяет CPU-аугментацию по процессам.
Используется детерминированное зерно на изображение; результаты двух реализаций сверяются.
Артефакты: `data/output/augmented.npy` и `data/output/performance_report.json`.
Для маленькой выборки параллельный режим может оказаться медленнее.
