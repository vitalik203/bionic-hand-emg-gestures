"""Емуляція потоку sEMG у реальному часі та каузальна фільтрація вікон зі станом.

* :class:`EMGStreamEmulator` програє запис ``[час, канали]`` з урахуванням ``fs``
  і віддає вікна (за замовчуванням 200 мс із кроком 50 мс) через генератор
  :meth:`~EMGStreamEmulator.windows` або через ``queue.Queue``
  (:meth:`~EMGStreamEmulator.start`). Підтримує паузу, зупинку та прискорення
  ``speed`` (у ``speed`` разів швидше за реальний час).
* :class:`CausalWindowFilter` фільтрує вікна, що перекриваються: перше вікно
  фільтрується повністю, далі — лише нові ``step`` відліків, решта береться з
  буфера вже відфільтрованих. Результат збігається з каузальним фільтром усього
  запису (:func:`emg_preprocess.causal_filter`).

Часова умова: відлік ``i`` покриває інтервал ``[i/fs, (i+1)/fs)`` і стає доступним
у момент ``(i+1)/fs`` від початку потоку. Тож вікно, що закінчується відліком
``end - 1``, готове в момент ``end / fs``.

Приклад::

    emu = EMGStreamEmulator(signals, fs=200.0, speed=10)
    for w in emu.windows():
        print(w.index, w.t_stream, w.data.shape)   # data: [40, 8]
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

from emg_preprocess import StreamingFilter


@dataclass(frozen=True)
class StreamWindow:
    """Одне вікно потоку.

    Attributes:
        index: порядковий номер вікна (0, 1, …).
        start: індекс першого відліку вікна в записі.
        end: індекс після останнього відліку (зріз ``[start:end]``).
        t_stream: час потоку, коли надійшов останній відлік вікна, с (``end / fs``).
        emitted_at: значення годинника емулятора в момент віддачі вікна.
        data: відліки ``[win, канали]``.
    """

    index: int
    start: int
    end: int
    t_stream: float
    emitted_at: float
    data: np.ndarray


class EMGStreamEmulator:
    """Програє запис як потік вікон у (прискореному) реальному часі."""

    def __init__(
        self,
        record: np.ndarray,
        fs: float,
        win_ms: float = 200.0,
        step_ms: float = 50.0,
        speed: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Args:
            record: запис ``[час, канали]``.
            fs: частота дискретизації запису, Гц.
            win_ms, step_ms: довжина вікна і крок, мс.
            speed: прискорення (10 — у 10 разів швидше за реальний час).
            clock, sleep: годинник і функція очікування (підмінюються в тестах).

        Raises:
            ValueError: при неправильній формі, fs/speed ≤ 0 або вікні, довшому за запис.
        """
        record = np.asarray(record)
        if record.ndim != 2:
            raise ValueError("record має форму [час, канали]")
        if fs <= 0 or speed <= 0:
            raise ValueError("fs і speed мають бути додатними")
        self.win = int(round(win_ms * fs / 1000.0))
        self.step = int(round(step_ms * fs / 1000.0))
        if self.win < 1 or self.step < 1:
            raise ValueError("вікно і крок мають містити щонайменше один відлік")
        if record.shape[0] < self.win:
            raise ValueError(f"запис ({record.shape[0]} відліків) коротший за вікно ({self.win})")
        self.record, self.fs, self.speed = record, float(fs), float(speed)
        self._clock, self._sleep = clock, sleep
        self._running = threading.Event()   # встановлено = не на паузі
        self._running.set()
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._paused_at: Optional[float] = None
        self._paused_total = 0.0

    @property
    def n_windows(self) -> int:
        """Кількість вікон, які дасть запис."""
        return (self.record.shape[0] - self.win) // self.step + 1

    @property
    def is_paused(self) -> bool:
        """Чи потік на паузі."""
        return not self._running.is_set()

    @property
    def is_stopped(self) -> bool:
        """Чи потік зупинено."""
        return self._stopped.is_set()

    def pause(self) -> None:
        """Ставить потік на паузу (час потоку зупиняється)."""
        if not self.is_paused:
            self._paused_at = self._clock()
            self._running.clear()

    def resume(self) -> None:
        """Продовжує потік після паузи."""
        if self.is_paused:
            self._paused_total += self._clock() - (self._paused_at or self._clock())
            self._paused_at = None
            self._running.set()

    def stop(self) -> None:
        """Зупиняє потік; генератор завершиться, у черзу піде ``None``."""
        self._stopped.set()
        self._running.set()  # розблокувати очікування на паузі

    def windows(self) -> Iterator[StreamWindow]:
        """Генератор вікон у реальному часі з урахуванням ``fs`` і ``speed``."""
        t0 = self._clock()
        self._paused_total = 0.0
        for k in range(self.n_windows):
            start = k * self.step
            end = start + self.win
            t_stream = end / self.fs
            while True:
                if self._stopped.is_set():
                    return
                if self.is_paused:
                    self._running.wait(timeout=0.05)
                    continue
                remaining = t0 + self._paused_total + t_stream / self.speed - self._clock()
                if remaining <= 0:
                    break
                self._sleep(min(remaining, 0.05))
            yield StreamWindow(k, start, end, t_stream, self._clock(), self.record[start:end])
        self._stopped.set()

    def start(self, maxsize: int = 0) -> "queue.Queue[Optional[StreamWindow]]":
        """Запускає потік у фоновому треді й повертає чергу вікон (``None`` — кінець)."""
        if self._thread is not None:
            raise RuntimeError("потік уже запущено")
        q: "queue.Queue[Optional[StreamWindow]]" = queue.Queue(maxsize=maxsize)

        def run() -> None:
            try:
                for w in self.windows():
                    q.put(w)
            finally:
                q.put(None)

        self._thread = threading.Thread(target=run, name="EMGStreamEmulator", daemon=True)
        self._thread.start()
        return q

    def join(self, timeout: Optional[float] = None) -> None:
        """Чекає завершення фонового треду (після :meth:`start`)."""
        if self._thread is not None:
            self._thread.join(timeout)


class CausalWindowFilter:
    """Каузальний фільтр для послідовних вікон, що перекриваються з кроком ``step``.

    Перше вікно фільтрується повністю (стан фільтра ініціалізується його першим
    відліком). Для кожного наступного вікна фільтруються лише останні ``step``
    відліків, а решта береться з буфера вже відфільтрованих. Вікна мають іти
    підряд: якщо вікно не продовжує попереднє, кидається ``ValueError``.
    """

    def __init__(self, fs: float, n_channels: int, win: int, step: int, **design_kw) -> None:
        """Args:
            fs: частота дискретизації, Гц.
            n_channels: кількість каналів.
            win, step: довжина вікна і крок у відліках.
            **design_kw: параметри :func:`emg_preprocess.design_sos`.
        """
        if not 1 <= step <= win:
            raise ValueError("потрібно 1 ≤ step ≤ win")
        self.win, self.step, self.n_channels = int(win), int(step), int(n_channels)
        self._filter = StreamingFilter(fs, n_channels, **design_kw)
        self._raw: Optional[np.ndarray] = None
        self._filtered: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Скидає стан (наступне вікно вважається першим)."""
        self._filter.reset()
        self._raw = self._filtered = None

    def process(self, window: np.ndarray) -> np.ndarray:
        """Повертає відфільтроване вікно ``[win, канали]`` (float64)."""
        w = np.asarray(window, dtype=np.float64)
        if w.shape != (self.win, self.n_channels):
            raise ValueError(f"очікувалось вікно {(self.win, self.n_channels)}, отримано {w.shape}")
        if self._raw is None:
            self._filtered = self._filter.process(w)
        else:
            if not np.array_equal(w[:-self.step], self._raw[self.step:]):
                raise ValueError("вікно не продовжує попереднє з кроком step; викличте reset()")
            new = self._filter.process(w[-self.step:])
            self._filtered = np.concatenate([self._filtered[self.step:], new])
        self._raw = w
        return self._filtered.copy()
