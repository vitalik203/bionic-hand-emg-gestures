"""Тести emg_data на синтетичних файлах у форматі UCI та (якщо є) на реальних даних."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import emg_data
from emg_data import (
    BOUNDARY_LABEL,
    load_subject,
    make_windows,
    read_raw_file,
    resample_uniform,
)

HEADER = "time\t" + "\t".join(f"channel{i}" for i in range(1, 9)) + "\tclass"


def _write_uci(path: Path, times, labels, values=None, extra_lines: str = "") -> None:
    """Пише файл у форматі UCI (TSV, CRLF); канал k має значення values[i] + k."""
    if values is None:
        values = np.arange(len(times), dtype=float)
    lines = [HEADER]
    for t, lab, v in zip(times, labels, values):
        chans = "\t".join(f"{(v + k) * 1e-5:g}" for k in range(8))
        lines.append(f"{t}\t{chans}\t{lab}")
    path.write_bytes(("\r\n".join(lines) + "\r\n" + extra_lines).encode("ascii"))


# ---------- read_raw_file ----------

def test_read_raw_file_parses_crlf_tsv(tmp_path: Path) -> None:
    p = tmp_path / "1_raw_data_x.txt"
    _write_uci(p, [1, 5, 6], [0, 1, 1])
    t, s, l = read_raw_file(p)
    np.testing.assert_array_equal(t, [1, 5, 6])
    assert s.shape == (3, 8) and s.dtype == np.float32
    np.testing.assert_allclose(s[1], (1 + np.arange(8)) * 1e-5, rtol=1e-5)
    np.testing.assert_array_equal(l, [0, 1, 1])
    assert l.dtype == np.int64


def test_read_raw_file_drops_truncated_row(tmp_path: Path) -> None:
    """Регресія: останній рядок файлу суб'єкта 34 має 9 стовпців замість 10."""
    p = tmp_path / "1_raw_data_x.txt"
    _write_uci(p, [1, 2, 3], [0, 0, 0], extra_lines="4\t1e-05\t1e-05\t1e-05\t1e-05\t1e-05\t1e-05\t0\t0")
    t, s, l = read_raw_file(p)
    assert len(t) == 3 and s.shape == (3, 8)


def test_read_raw_file_sorts_and_dedups_time(tmp_path: Path) -> None:
    p = tmp_path / "1_raw_data_x.txt"
    _write_uci(p, [3, 1, 1, 2], [0, 1, 2, 0], values=[30, 10, 11, 20])
    t, s, l = read_raw_file(p)
    np.testing.assert_array_equal(t, [1, 2, 3])
    np.testing.assert_array_equal(l, [1, 0, 0])


def test_read_raw_file_rejects_bad_header(tmp_path: Path) -> None:
    p = tmp_path / "bad.txt"
    p.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError):
        read_raw_file(p)


# ---------- resample_uniform ----------

def test_resample_holds_previous_value() -> None:
    t = np.array([0.0, 1.0, 3.0, 4.0])
    s = np.array([[0.0], [1.0], [3.0], [4.0]], dtype=np.float32)
    l = np.array([0, 1, 1, 2])
    s_u, l_u = resample_uniform(t, s, l, fs=1000.0)
    np.testing.assert_array_equal(s_u[:, 0], [0, 1, 1, 3, 4])
    np.testing.assert_array_equal(l_u, [0, 1, 1, 1, 2])


def test_resample_marks_long_gap_as_boundary() -> None:
    """Регресія: у суб'єкта 34 є пропуск часу 13,8 с — не можна заповнювати його сигналом."""
    t = np.array([0.0, 1.0, 500.0, 501.0])
    s = np.zeros((4, 8), dtype=np.float32)
    l = np.array([1, 1, 1, 1])
    _, l_u = resample_uniform(t, s, l, fs=1000.0, max_gap_ms=100.0)
    assert len(l_u) == 502
    assert l_u[0] == 1 and l_u[1] == 1 and l_u[500] == 1 and l_u[501] == 1
    assert np.all(l_u[2:500] == BOUNDARY_LABEL)


def test_resample_short_gap_is_filled() -> None:
    t = np.array([0.0, 30.0])
    _, l_u = resample_uniform(t, np.zeros((2, 1)), np.array([2, 2]), fs=1000.0, max_gap_ms=100.0)
    assert np.all(l_u == 2)


def test_resample_rejects_non_monotonic_time() -> None:
    with pytest.raises(ValueError):
        resample_uniform(np.array([0.0, 0.0]), np.zeros((2, 1)), np.array([0, 0]))


# ---------- load_subject ----------

def test_load_subject_concatenates_series_with_boundary(tmp_path: Path) -> None:
    d = tmp_path / "07"
    d.mkdir()
    _write_uci(d / "2_raw_data_b.txt", range(1, 4), [2, 2, 2])
    _write_uci(d / "1_raw_data_a.txt", range(1, 6), [1, 1, 1, 1, 1])
    s, l = load_subject(7, data_dir=tmp_path, fs=1000.0)
    assert s.shape == (5 + 1 + 3, 8) and s.dtype == np.float32
    np.testing.assert_array_equal(l, [1, 1, 1, 1, 1, BOUNDARY_LABEL, 2, 2, 2])


def test_load_subject_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_subject(99, data_dir=tmp_path)


def test_window_duration_follows_timestamps_not_row_count(tmp_path: Path) -> None:
    """Регресія: рядки йдуть ~1 кГц, а не 200 Гц. 200-мс вікно має охоплювати 200 мс часу."""
    d = tmp_path / "01"
    d.mkdir()
    times = np.arange(1, 2001)  # 2 с з кроком 1 мс
    _write_uci(d / "1_raw_data_a.txt", times, [1] * 2000, values=times.astype(float))
    s, l = load_subject(1, data_dir=tmp_path, fs=1000.0)
    X, _ = make_windows(s, l, fs=1000.0, win_ms=200, step_ms=50)
    span_ms = (X[:, -1, 0] - X[:, 0, 0]) / 1e-5 + 1  # значення каналу 0 = час у мс
    np.testing.assert_allclose(span_ms, 200.0, atol=1e-2)


# ---------- make_windows ----------

def _toy(n: int = 40, c: int = 3):
    s = np.arange(n * c, dtype=np.float32).reshape(n, c)
    l = np.array([1] * 10 + [2] * 10 + [0] * 10 + [3] * 10)[:n]
    return s, l


def test_make_windows_shapes_and_content() -> None:
    s, l = _toy()
    X, y = make_windows(s, l, fs=100.0, win_ms=50, step_ms=50)  # win=5, step=5
    assert X.shape == (8, 5, 3) and y.shape == (8,)
    np.testing.assert_array_equal(X[0], s[0:5])
    np.testing.assert_array_equal(y, [1, 1, 2, 2, 0, 0, 3, 3])


def test_make_windows_drops_transitions() -> None:
    s, l = _toy()
    X, y = make_windows(s, l, fs=100.0, win_ms=40, step_ms=30)  # win=4, step=3
    starts = np.arange(0, 40 - 4 + 1, 3)
    expected = [l[a] for a in starts if len(set(l[a:a + 4])) == 1]
    np.testing.assert_array_equal(y, expected)
    assert len(y) < len(starts)


def test_make_windows_ignore_labels() -> None:
    s, l = _toy()
    _, y = make_windows(s, l, fs=100.0, win_ms=50, step_ms=50, ignore_labels=(0,))
    assert 0 not in y and set(y) == {1, 2, 3}


def test_make_windows_majority_when_transitions_kept() -> None:
    s = np.zeros((6, 1), dtype=np.float32)
    l = np.array([1, 1, 2, 2, 2, 2])
    _, y = make_windows(s, l, fs=1000.0, win_ms=3, step_ms=1, drop_transitions=False)
    np.testing.assert_array_equal(y, [1, 2, 2, 2])  # [1,1,2] -> 1, [1,2,2] -> 2


def test_make_windows_always_drops_boundary() -> None:
    s = np.zeros((9, 1), dtype=np.float32)
    l = np.array([1, 1, 1, 1, BOUNDARY_LABEL, 1, 1, 1, 1])
    _, y = make_windows(s, l, fs=1000.0, win_ms=3, step_ms=1, drop_transitions=False)
    assert len(y) == 4  # лише вікна, що не зачіпають індекс 4


def test_make_windows_short_signal_returns_empty() -> None:
    X, y = make_windows(np.zeros((3, 8)), np.zeros(3, dtype=int), fs=1000.0, win_ms=200)
    assert X.shape == (0, 200, 8) and y.shape == (0,)


@pytest.mark.parametrize(
    "signals,labels,kw",
    [
        (np.zeros(10), np.zeros(10), {}),
        (np.zeros((10, 2)), np.zeros(9), {}),
        (np.zeros((10, 2)), np.zeros(10), {"win_ms": 0.1}),
    ],
)
def test_make_windows_validation(signals, labels, kw) -> None:
    with pytest.raises(ValueError):
        make_windows(signals, labels, fs=1000.0, **kw)


# ---------- реальні дані (пропускаються, якщо датасету немає) ----------

_REAL = emg_data.DEFAULT_DATA_DIR
_needs_real = pytest.mark.skipif(not (_REAL / "01").is_dir(), reason="немає data/EMG_data_for_gestures-master")


@_needs_real
def test_real_subject_shapes() -> None:
    s, l = load_subject(1)
    assert s.shape[1] == 8 and s.shape[0] == l.shape[0]
    assert set(np.unique(l)) <= {BOUNDARY_LABEL, 0, 1, 2, 3, 4, 5, 6, 7}


@_needs_real
@pytest.mark.parametrize("subject,has7", [(11, True), (30, True), (1, False), (34, False)])
def test_real_class7_presence(subject: int, has7: bool) -> None:
    _, l = load_subject(subject)
    assert (7 in l) == has7
