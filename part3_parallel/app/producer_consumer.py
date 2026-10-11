"""Производитель и потребители с двумя потокобезопасными очередями."""

import logging
import queue
import threading
from pathlib import Path

from part2_prepare.app.scanner import list_matching_files, read_image

LOGGER = logging.getLogger(__name__)


class ImagePipeline:
    """Сканирует файлы в одном потоке, читает в потоках-потребителях."""

    def __init__(self, workers=3, size=32):
        if workers < 1:
            raise ValueError('Нужно не менее одного потока-потребителя.')
        self.workers = workers
        self.size = size
        self.count = 0
        self.counter_lock = threading.Lock()

    def load(self, directory, extensions='.png,.jpg,.jpeg'):
        """Загружает изображения с корректной остановкой каждого рабочего потока."""
        paths = []
        file_queue = queue.Queue(maxsize=128)
        result_queue = queue.Queue()
        stop = object()
        results = {}
        errors = []
        self.count = 0

        def producer():
            try:
                paths.extend(list_matching_files(directory, extensions))
                LOGGER.info('Producer: найдено файлов: %d', len(paths))
                for index, path in enumerate(paths):
                    file_queue.put((index, path))
            except Exception as error:
                errors.append(('producer', str(error)))
            finally:
                for _ in range(self.workers):
                    file_queue.put(stop)

        def consumer(worker_index):
            while True:
                task = file_queue.get()
                try:
                    if task is stop:
                        LOGGER.info('Consumer-%d: завершение', worker_index)
                        return
                    index, path = task
                    try:
                        image = read_image(path, self.size)
                        result_queue.put((index, image))
                        with self.counter_lock:
                            self.count += 1
                        LOGGER.info('Consumer-%d: %s', worker_index, Path(path).name)
                    except (OSError, ValueError) as error:
                        errors.append((path, str(error)))
                finally:
                    file_queue.task_done()

        producer_thread = threading.Thread(target=producer, name='producer')
        consumer_threads = [threading.Thread(target=consumer, args=(index + 1,), name=f'consumer-{index + 1}') for index in range(self.workers)]
        for worker in consumer_threads:
            worker.start()
        producer_thread.start()
        producer_thread.join()
        file_queue.join()
        for worker in consumer_threads:
            worker.join()
        while not result_queue.empty():
            index, image = result_queue.get_nowait()
            results[index] = image
        if errors:
            raise RuntimeError(f'Ошибки чтения: {errors[:3]}')
        if len(results) != len(paths):
            raise RuntimeError('Не все файлы были получены из очереди.')
        return [results[index] for index in range(len(paths))]
