"""Сравнение последовательного и параллельного конвейера изображений."""

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from part2_prepare.app.scanner import list_matching_files, read_image
from .augment import augment_image, pool_augment
from .producer_consumer import ImagePipeline

LOGGER = logging.getLogger(__name__)


def run_sequential(paths, size, copies, seed):
    """Читает и аугментирует изображения в одном процессе."""
    arrays = [augment_image(index, read_image(path, size), copies, seed) for index, path in enumerate(paths)]
    return np.concatenate(arrays, axis=0)


def run_parallel(directory, paths, size, copies, seed, threads, processes):
    """Читает файлы в потоках и выполняет CPU-аугментацию в процессах."""
    images = ImagePipeline(threads, size).load(directory)
    tasks = [(index, image, copies, seed) for index, image in enumerate(images)]
    results = {}
    with mp.Pool(processes=processes) as pool:
        pending = [pool.apply_async(pool_augment, (task,)) for task in tasks]
        for operation in pending:
            index, batch = operation.get(timeout=90)
            results[index] = batch
    return np.concatenate([results[index] for index in range(len(paths))], axis=0)


def main(argv=None):
    """Выполняет оба режима, сверяет массивы и сохраняет JSON-отчёт."""
    parser = argparse.ArgumentParser(description='Сравнение CPU-аугментации')
    parser.add_argument('--path', default=os.getenv('NN_INPUT_PATH', 'data/input/images'))
    parser.add_argument('--output', default=os.getenv('NN_PARALLEL_OUTPUT', 'data/output/augmented.npy'))
    parser.add_argument('--report', default=os.getenv('NN_REPORT_PATH', 'data/output/performance_report.json'))
    parser.add_argument('--size', type=int, default=int(os.getenv('NN_IMAGE_SIZE', '32')))
    parser.add_argument('--copies', type=int, default=2)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args(argv)
    if min(args.size, args.threads, args.processes) < 1 or args.copies < 1:
        parser.error('Размер, количество потоков/процессов и копий должны быть положительными.')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    paths = list_matching_files(args.path, '.png,.jpg,.jpeg')
    print(f'Найдено изображений: {len(paths)}', flush=True)
    start = time.perf_counter()
    sequential = run_sequential(paths, args.size, args.copies, args.seed)
    seconds_sequential = time.perf_counter() - start
    print(f'Последовательная обработка: {seconds_sequential:.4f} с', flush=True)
    start = time.perf_counter()
    parallel = run_parallel(args.path, paths, args.size, args.copies, args.seed, args.threads, args.processes)
    seconds_parallel = time.perf_counter() - start
    if not np.array_equal(sequential, parallel):
        raise RuntimeError('Результаты последовательного и параллельного режимов различаются.')
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, parallel, allow_pickle=False)
    report = {
        'files': len(paths), 'result_samples': int(len(parallel)),
        'shape': list(parallel.shape), 'threads': args.threads, 'processes': args.processes,
        'augmentation_copies_per_image': args.copies, 'seed': args.seed,
        'sequential_seconds': round(seconds_sequential, 5),
        'parallel_seconds': round(seconds_parallel, 5),
        'speedup_ratio': round(seconds_sequential / seconds_parallel, 3),
        'results_identical': True,
        'note': 'Для малого набора данных запуск процессов может быть медленнее последовательного алгоритма.'
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Параллельная обработка: {seconds_parallel:.4f} с', flush=True)
    print(f'Отношение T_seq / T_parallel: {report["speedup_ratio"]:.3f}', flush=True)
    print(f'Массивы совпали: {report["results_identical"]}', flush=True)
    print(f'Результат: {destination}', flush=True)
    print(f'Отчёт: {report_path}', flush=True)
    return 0


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
