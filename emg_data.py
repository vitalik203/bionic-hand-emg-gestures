"""Завантаження UCI «EMG data for gestures» (id 481) і нарізка на вікна.

Факти про файли, перевірені на локальній копії (див. docs/datasets.md):

* кожен файл — TSV із заголовком ``time, channel1..channel8, class``, рядки CRLF;
* стовпець ``time`` — мілісекунди, крок нерегулярний (медіана 1 мс,
  ~820–970 рядків/с), тому частоту дискретизації НЕ можна брати як константу
  «200 Гц»: сигнал перед нарізкою переводиться на рівномірну сітку за часом;
* значення каналів змінюються лише ~100 разів/с, між змінами рядки
  повторюються (sample-and-hold), тож на сітці 1 кГц вікно 200 мс містить
  ~20 різних відліків на канал;
* трапляються пропуски часу (у суб'єкта 34 — 13,8 с) і обрізаний останній
  рядок (9 стовпців замість 10).

Ділянки, яких немає в записі (пропуски, межі між серіями), позначаються
міткою :data:`BOUNDARY_LABEL`; вікна з такими відліками завжди відкидаються.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PathLike = Union[str, Path]

#: Кількість ЕМГ-каналів браслета Myo у датасеті.
UCI_N_CHANNELS: int = 8
#: Частота рівномірної сітки за замовчуванням, Гц (крок 1 мс збігається з
#: роздільністю стовпця ``time``).
UCI_FS_HZ: float = 1000.0
#: Службова мітка для відліків поза записом (пропуски, межі серій).
BOUNDARY_LABEL: int = -1
#: Назви класів за README датасету.
UCI_CLASS_NAMES: Dict[int, str] = {
    0: "без розмітки",
    1: "спокій",
    2: "кулак",
    3: "згинання зап'ястя",
    4: "розгинання зап'ястя",
    5: "радіальне відведення",
    6: "ульнарне відведення",
    7: "розкрита долоня",
}
#: Тека з розпакованим архівом датасету.
DEFAULT_DATA_DIR: Path = Path(__file__).resolve().parent / "data" / "EMG_data_for_gestures-master"

_N_COLUMNS = 1 + UCI_N_CHANNELS + 1


def read_raw_file(path: PathLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Читає один файл ``*_raw_data_*.txt`` без зміни часової сітки.

    Рядки з неправильною кількістю стовпців або нечисловими значеннями
    відкидаються. Рядки сортуються за часом, повтори часу прибираються
    (лишається перший).

    Args:
        path: шлях до файлу.

    Returns:
        ``(time_ms [T] float64, signals [T, 8] float32, labels [T] int64)``.

    Raises:
        ValueError: якщо заголовок не відповідає формату UCI, немає жодного
            коректного рядка або мітки не цілі.
    """
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        if len(header) != _N_COLUMNS or header[0] != "time" or header[-1] != "class":
            raise ValueError(f"{path}: неочікуваний заголовок {header}")
        rows = [parts for parts in (line.split() for line in f) if len(parts) == _N_COLUMNS]
    if not rows:
        raise ValueError(f"{path}: немає жодного коректного рядка")

    data = np.asarray(rows, dtype=np.float64)
    data = data[np.isfinite(data).all(axis=1)]
    data = data[np.argsort(data[:, 0], kind="stable")]
    _, first = np.unique(data[:, 0], return_index=True)
    data = data[first]

    labels_f = data[:, -1]
    if not np.all(labels_f == np.round(labels_f)):
        raise ValueError(f"{path}: мітки класів не цілі")
    return data[:, 0], data[:, 1:-1].astype(np.float32), labels_f.astype(np.int64)


def resample_uniform(
    time_ms: np.ndarray,
    signals: np.ndarray,
    labels: np.ndarray,
    fs: float = UCI_FS_HZ,
    max_gap_ms: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Переводить запис на рівномірну сітку методом утримання попереднього значення.

    Утримання (а не інтерполяція) відповідає тому, як дані вже записані у
    файлі: значення між оновленнями повторюються. Відліки сітки всередині
    пропуску, довшого за ``max_gap_ms``, отримують мітку :data:`BOUNDARY_LABEL`.

    Args:
        time_ms: час відліків у мс, строго зростає, форма ``[T]``.
        signals: сигнали ``[T, C]``.
        labels: мітки ``[T]``.
        fs: частота сітки, Гц.
        max_gap_ms: найбільший пропуск, який ще заповнюється утриманням.

    Returns:
        ``(signals [M, C], labels [M])`` на сітці ``time_ms[0] + k * 1000 / fs``.

    Raises:
        ValueError: при невідповідності форм, порожньому записі, fs <= 0 або
            нестрого зростаючому часі.
    """
    time_ms = np.asarray(time_ms, dtype=np.float64)
    if fs <= 0:
        raise ValueError("fs має бути додатною")
    if time_ms.ndim != 1 or len(time_ms) == 0:
        raise ValueError("time_ms має бути непорожнім одновимірним масивом")
    if signals.shape[0] != len(time_ms) or labels.shape[0] != len(time_ms):
        raise ValueError("time_ms, signals і labels мають однакову довжину по осі часу")
    if np.any(np.diff(time_ms) <= 0):
        raise ValueError("time_ms має строго зростати")

    step = 1000.0 / fs
    n = int(np.floor((time_ms[-1] - time_ms[0]) / step + 1e-9)) + 1
    grid = time_ms[0] + np.arange(n) * step
    idx = np.searchsorted(time_ms, grid + 1e-9, side="right") - 1

    out_labels = labels[idx].astype(np.int64, copy=True)
    gap_after = np.append(np.diff(time_ms), 0.0)
    in_gap = (gap_after[idx] > max_gap_ms) & (grid > time_ms[idx] + 1e-9)
    out_labels[in_gap] = BOUNDARY_LABEL
    return signals[idx], out_labels


def subject_files(subject_id: int, data_dir: PathLike = DEFAULT_DATA_DIR) -> List[Path]:
    """Повертає файли серій піддослідного, впорядковані за номером серії.

    Args:
        subject_id: номер піддослідного (1–36 у повному датасеті).
        data_dir: тека з підтеками ``01`` … ``36``.

    Returns:
        Список :class:`pathlib.Path`.

    Raises:
        FileNotFoundError: якщо теки піддослідного немає або в ній немає файлів.
    """
    subj_dir = Path(data_dir) / f"{int(subject_id):02d}"
    files = sorted(subj_dir.glob("*_raw_data_*.txt"), key=lambda p: (int(p.name.split("_")[0]), p.name))
    if not files:
        raise FileNotFoundError(f"Немає файлів *_raw_data_*.txt у {subj_dir}")
    return files


def load_subject(
    subject_id: int,
    data_dir: PathLike = DEFAULT_DATA_DIR,
    fs: float = UCI_FS_HZ,
    max_gap_ms: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Завантажує всі серії піддослідного на рівномірній сітці ``fs``.

    Серії склеюються послідовно; між ними вставляється один відлік із
    міткою :data:`BOUNDARY_LABEL` (сигнал — нулі), щоб жодне вікно не
    перетинало межу серій.

    Args:
        subject_id: номер піддослідного.
        data_dir: тека датасету.
        fs: частота рівномірної сітки, Гц. Те саме значення слід передати в
            :func:`make_windows`.
        max_gap_ms: див. :func:`resample_uniform`.

    Returns:
        ``(signals [T, 8] float32, labels [T] int64)``.
    """
    sig_parts, lab_parts = [], []
    for i, (s_u, l_u) in enumerate(load_subject_series(subject_id, data_dir, fs, max_gap_ms)):
        if i > 0:
            sig_parts.append(np.zeros((1, s_u.shape[1]), dtype=np.float32))
            lab_parts.append(np.array([BOUNDARY_LABEL], dtype=np.int64))
        sig_parts.append(s_u)
        lab_parts.append(l_u)
    return np.concatenate(sig_parts, axis=0), np.concatenate(lab_parts)


def load_subject_series(
    subject_id: int,
    data_dir: PathLike = DEFAULT_DATA_DIR,
    fs: float = UCI_FS_HZ,
    max_gap_ms: float = 100.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Завантажує серії піддослідного окремо, кожну на рівномірній сітці ``fs``.

    На відміну від :func:`load_subject`, серії не склеюються. Усередині серії
    довгі пропуски все одно позначаються :data:`BOUNDARY_LABEL` (див.
    :func:`resample_uniform`), тож фільтрувати серію слід через
    :func:`emg_preprocess.preprocess_segments`.

    Args:
        subject_id: номер піддослідного.
        data_dir: тека датасету.
        fs: частота рівномірної сітки, Гц.
        max_gap_ms: див. :func:`resample_uniform`.

    Returns:
        Список ``(signals [T, 8] float32, labels [T] int64)`` у порядку серій.
    """
    series = []
    for path in subject_files(subject_id, data_dir):
        t, s, l = read_raw_file(path)
        s_u, l_u = resample_uniform(t, s, l, fs=fs, max_gap_ms=max_gap_ms)
        series.append((s_u.astype(np.float32, copy=False), l_u))
    return series


def make_windows(
    signals: np.ndarray,
    labels: np.ndarray,
    fs: float,
    win_ms: float = 200.0,
    step_ms: float = 50.0,
    ignore_labels: Iterable[int] = (),
    drop_transitions: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Нарізає рівномірний запис на вікна з перекриттям.

    Мітка вікна — найчастіша мітка його відліків (при рівності — менша).
    За замовчуванням вікна, у яких мітка змінюється, відкидаються; тоді мітка
    вікна просто збігається з міткою всіх його відліків. Вікна з відліками
    :data:`BOUNDARY_LABEL` відкидаються завжди.

    Args:
        signals: ``[T, C]`` на рівномірній сітці з частотою ``fs``.
        labels: ``[T]`` цілі мітки.
        fs: частота дискретизації, Гц.
        win_ms: довжина вікна, мс.
        step_ms: крок між початками вікон, мс.
        ignore_labels: мітки, вікна з якими відкидаються (напр. ``(0,)``).
        drop_transitions: відкидати вікна зі зміною мітки.

    Returns:
        ``(X [N, win, C], y [N] int64)``, де ``win = round(win_ms * fs / 1000)``.

    Raises:
        ValueError: при невідповідності форм або вікні/кроці коротшому за відлік.
    """
    signals = np.asarray(signals)
    labels = np.asarray(labels)
    if signals.ndim != 2:
        raise ValueError("signals має форму [час, канали]")
    if labels.ndim != 1 or labels.shape[0] != signals.shape[0]:
        raise ValueError("labels має форму [час] і ту саму довжину, що signals")
    if fs <= 0:
        raise ValueError("fs має бути додатною")
    win = int(round(win_ms * fs / 1000.0))
    step = int(round(step_ms * fs / 1000.0))
    if win < 1 or step < 1:
        raise ValueError("win_ms і step_ms мають давати щонайменше один відлік")

    n_time, n_ch = signals.shape
    if n_time < win:
        return np.empty((0, win, n_ch), dtype=signals.dtype), np.empty(0, dtype=np.int64)

    starts = np.arange(0, n_time - win + 1, step)
    lw = sliding_window_view(labels, win)[starts]  # [N, win]

    if drop_transitions:
        keep = np.all(lw == lw[:, :1], axis=1)
        y = lw[:, 0].astype(np.int64)
    else:
        classes = np.unique(labels)
        counts = (lw[:, :, None] == classes[None, None, :]).sum(axis=1)
        y = classes[np.argmax(counts, axis=1)].astype(np.int64)
        keep = np.ones(len(starts), dtype=bool)

    keep &= ~np.any(lw == BOUNDARY_LABEL, axis=1)
    ignore = np.asarray(list(ignore_labels), dtype=np.int64)
    if ignore.size:
        keep &= ~np.isin(y, ignore)

    sw = sliding_window_view(signals, win, axis=0)  # [T-win+1, C, win]
    X = np.ascontiguousarray(sw[starts[keep]].transpose(0, 2, 1))
    return X, y[keep]
