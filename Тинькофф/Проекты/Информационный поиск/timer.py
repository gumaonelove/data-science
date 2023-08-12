import time


class TimerError(Exception):
    """Пользовательское исключение, используемое для сообщения об ошибках при использовании класса Timer"""


class Timer:

    def __init__(self):
        self._start_time = None

    def start(self, text: str) -> None:
        """Запуск нового таймера"""

        if self._start_time is not None:
            raise TimerError(f"Таймер уже работает. Используйте .stop() чтобы его остановить")
        print(text)
        self._start_time = time.perf_counter()

    def stop(self, text: str) -> None:
        """Отстановить таймер и сообщить о времени вычисления"""

        if self._start_time is None:
            raise TimerError(f"Таймер не работает. Используйте .start() для его запуска")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"{text} {elapsed_time:0.4f} секунд")
