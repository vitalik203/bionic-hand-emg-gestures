"""Попередня обробка sEMG: смуговий і режекторний фільтри, потоковий фільтр, нормалізація.

Порядок роботи з безперервною серією::

    filtered = preprocess(signals, fs)            # нуль-фазова фільтрація всієї серії
    X, y = make_windows(filtered, labels, fs)     # нарізка ПІСЛЯ фільтрації
    norm = Normalizer().fit(train_signals)        # статистики лише з train
    X_train = norm.transform(X_train)

Для записів із :func:`emg_data.load_subject`, де серії склеєні через відліки
:data:`emg_data.BOUNDARY_LABEL`, використовуйте :func:`preprocess_segments`:
вона фільтрує кожну безперервну ділянку окремо.

Для потоку в реальному часі використовуйте :class:`StreamingFilter`: це
каузальний фільтр зі збереженням стану між блоками.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal as sps

from emg_data import BOUNDARY_LABEL

PathLike = Union[str, Path]

#: Нижня межа смуги пропускання, Гц.
LOW_HZ: float = 20.0
#: Верхня межа смуги пропускання, Гц (обмежується ``HIGH_FRAC * fs``).
HIGH_HZ: float = 450.0
#: Частка fs, вище якої верхня межа не піднімається.
HIGH_FRAC: float = 0.45
#: Порядок прототипу Баттерворта (для смугового фільтра це 4 біквадратні секції).
ORDER: int = 4
#: Частота мережевої завади, Гц.
NOTCH_HZ: float = 50.0
#: Добротність режекторного фільтра.
NOTCH_Q: float = 30.0


def band_edges(fs: float, low_hz: float = LOW_HZ, high_hz: float = HIGH_HZ) -> Tuple[float, float]:
    """Межі смуги пропускання ``(low, min(high, 0.45 * fs))``.

    Args:
        fs: частота дискретизації, Гц.
        low_hz: нижня межа, Гц.
        high_hz: бажана верхня межа, Гц.

    Returns:
        ``(low, high)`` у Гц.

    Raises:
        ValueError: якщо fs <= 0 або смуга порожня (наприклад, fs < 20 / 0.45 ≈ 44,4 Гц).
    """
    if fs <= 0:
        raise ValueError("fs має бути додатною")
    high = min(high_hz, HIGH_FRAC * fs)
    if not 0 < low_hz < high:
        raise ValueError(f"порожня смуга: low={low_hz} Гц, high={high:.2f} Гц при fs={fs} Гц")
    return float(low_hz), float(high)


def design_sos(
    fs: float,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
    order: int = ORDER,
    notch_hz: Optional[float] = NOTCH_HZ,
    notch_q: float = NOTCH_Q,
) -> np.ndarray:
    """Каскад секцій другого порядку: смуговий Баттерворт + режекторний фільтр.

    ``scipy.signal.butter(order, ..., btype="bandpass")`` дає фільтр загального
    порядку ``2 * order``, тобто ``order`` секцій. Режекторний фільтр додається
    однією секцією. Якщо ``notch_hz >= fs / 2``, такої частоти в сигналі бути не
    може, і режекторний фільтр пропускається.

    Args:
        fs: частота дискретизації, Гц.
        low_hz, high_hz: межі смуги (див. :func:`band_edges`).
        order: порядок прототипу Баттерворта.
        notch_hz: частота режекції, Гц; ``None`` — без режекції.
        notch_q: добротність режекторного фільтра.

    Returns:
        Масив sos форми ``[n_sections, 6]``.
    """
    low, high = band_edges(fs, low_hz, high_hz)
    sos = sps.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    if notch_hz is not None and 0 < notch_hz < fs / 2:
        b, a = sps.iirnotch(notch_hz, notch_q, fs=fs)
        sos = np.vstack([sos, sps.tf2sos(b, a)])
    return sos


def _check_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal)
    if x.ndim != 2:
        raise ValueError("signal має форму [час, канали]")
    return x


def _out_dtype(x: np.ndarray) -> np.dtype:
    return x.dtype if np.issubdtype(x.dtype, np.floating) else np.dtype(np.float64)


def preprocess(signal: np.ndarray, fs: float, **design_kw) -> np.ndarray:
    """Нуль-фазова фільтрація безперервної серії (``sosfiltfilt`` по осі часу).

    Фільтр застосовується двічі (вперед і назад), тож фазового зсуву немає, а
    АЧХ — квадрат АЧХ :func:`design_sos`. Це офлайн-варіант для навчання;
    у потоці так зробити неможливо, бо потрібні майбутні відліки.

    Args:
        signal: ``[час, канали]``, одна безперервна серія.
        fs: частота дискретизації, Гц.
        **design_kw: параметри :func:`design_sos`.

    Returns:
        Відфільтрований сигнал тієї ж форми; float-тип входу зберігається.

    Raises:
        ValueError: при неправильній формі або занадто короткому сигналі.
    """
    x = _check_signal(signal)
    sos = design_sos(fs, **design_kw)
    try:
        y = sps.sosfiltfilt(sos, x.astype(np.float64), axis=0)
    except ValueError as exc:
        raise ValueError(f"сигнал із {x.shape[0]} відліків закороткий для sosfiltfilt: {exc}") from exc
    return y.astype(_out_dtype(x), copy=False)


def preprocess_segments(signals: np.ndarray, labels: np.ndarray, fs: float, **design_kw) -> np.ndarray:
    """Фільтрує окремо кожну безперервну ділянку між відліками ``BOUNDARY_LABEL``.

    Відліки з міткою :data:`emg_data.BOUNDARY_LABEL` у результаті дорівнюють
    нулю. Ці вікна однаково відкидає :func:`emg_data.make_windows`.

    Args:
        signals: ``[час, канали]`` із :func:`emg_data.load_subject`.
        labels: ``[час]``.
        fs: частота дискретизації, Гц.
        **design_kw: параметри :func:`design_sos`.

    Returns:
        Відфільтрований сигнал тієї ж форми.
    """
    x = _check_signal(signals)
    labels = np.asarray(labels)
    if labels.shape != (x.shape[0],):
        raise ValueError("labels має форму [час] і ту саму довжину, що signals")
    out = np.zeros(x.shape, dtype=_out_dtype(x))
    valid = labels != BOUNDARY_LABEL
    edges = np.flatnonzero(np.diff(np.concatenate([[0], valid.astype(np.int8), [0]])))
    for start, stop in zip(edges[::2], edges[1::2]):
        out[start:stop] = preprocess(x[start:stop], fs, **design_kw)
    return out


class StreamingFilter:
    """Каузальний фільтр для потоку: ``sosfilt`` зі збереженням стану між блоками.

    Результат обробки блоками будь-якої довжини збігається з :func:`causal_filter`
    на всьому сигналі (з точністю до округлення). Початковий стан —
    ``sosfilt_zi * перший_відлік``, щоб зменшити перехідний процес на старті.

    Приклад::

        sf = StreamingFilter(fs=1000.0, n_channels=8)
        for block in blocks:            # block: [n, 8]
            y = sf.process(block)
    """

    def __init__(self, fs: float, n_channels: int, **design_kw) -> None:
        """Args:
            fs: частота дискретизації, Гц.
            n_channels: кількість каналів.
            **design_kw: параметри :func:`design_sos`.
        """
        self.fs = float(fs)
        self.n_channels = int(n_channels)
        self.sos = design_sos(fs, **design_kw)
        self._zi: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Скидає стан фільтра (наступний блок почнеться як новий запис)."""
        self._zi = None

    def process(self, block: np.ndarray) -> np.ndarray:
        """Фільтрує черговий блок ``[n, канали]`` і оновлює стан.

        Returns:
            Відфільтрований блок ``[n, канали]`` (float64).
        """
        x = _check_signal(block).astype(np.float64)
        if x.shape[1] != self.n_channels:
            raise ValueError(f"очікувалось {self.n_channels} каналів, отримано {x.shape[1]}")
        if x.shape[0] == 0:
            return x.copy()
        if self._zi is None:
            self._zi = sps.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self._zi = sps.sosfilt(self.sos, x, axis=0, zi=self._zi)
        return y


def causal_filter(signal: np.ndarray, fs: float, **design_kw) -> np.ndarray:
    """Офлайн-еквівалент :class:`StreamingFilter` на всьому сигналі одразу.

    Returns:
        Відфільтрований сигнал ``[час, канали]`` (float64).
    """
    x = _check_signal(signal)
    sf = StreamingFilter(fs, x.shape[1], **design_kw)
    return sf.process(x)


class Normalizer:
    """Z-нормалізація по каналу: статистики вчаться лише на train.

    ``fit`` приймає ``[час, канали]`` або ``[N, час, канали]``; середнє й
    стандартне відхилення рахуються по всіх осях, крім останньої. Канали з
    нульовим розкидом отримують std = 1 (без ділення на нуль).

    Рекомендація: викликайте ``fit`` на безперервних train-сигналах, а не на
    вікнах. Вікна з кроком 50 мс і довжиною 200 мс перекриваються, тож кожен
    відлік увійшов би в статистику ~4 рази.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        """Args:
            eps: std, менші за це значення, замінюються на 1.
        """
        self.eps = float(eps)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        """Чи вже обчислено статистики."""
        return self.mean_ is not None

    def fit(self, X: np.ndarray) -> "Normalizer":
        """Обчислює mean/std по каналу на train-даних.

        Returns:
            ``self``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim < 2 or X.size == 0:
            raise ValueError("X має форму [..., канали] і не може бути порожнім")
        flat = X.reshape(-1, X.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < self.eps] = 1.0
        self.mean_, self.std_ = mean, std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Застосовує збережені статистики: ``(X - mean) / std``."""
        if not self.is_fitted:
            raise RuntimeError("спочатку викличте fit на train-даних")
        X = np.asarray(X)
        if X.shape[-1] != self.mean_.shape[0]:
            raise ValueError(f"очікувалось {self.mean_.shape[0]} каналів, отримано {X.shape[-1]}")
        out = (X.astype(np.float64) - self.mean_) / self.std_
        return out.astype(_out_dtype(X), copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """``fit(X)``, потім ``transform(X)``; лише для train."""
        return self.fit(X).transform(X)

    def save(self, path: PathLike) -> None:
        """Зберігає статистики в ``.npz``."""
        if not self.is_fitted:
            raise RuntimeError("нема чого зберігати: Normalizer не навчено")
        np.savez(path, mean=self.mean_, std=self.std_, eps=self.eps)

    @classmethod
    def load(cls, path: PathLike) -> "Normalizer":
        """Завантажує статистики, збережені :meth:`save`."""
        with np.load(path) as data:
            norm = cls(eps=float(data["eps"]))
            norm.mean_ = data["mean"].astype(np.float64)
            norm.std_ = data["std"].astype(np.float64)
        return norm
