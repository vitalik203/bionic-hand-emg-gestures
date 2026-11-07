"""Тести допоміжних функцій scripts/plot_signals.py (без малювання)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "plot_signals", Path(__file__).resolve().parents[1] / "scripts" / "plot_signals.py"
)
plot_signals = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_signals)


def test_label_segments() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0, 2])
    assert plot_signals.label_segments(labels) == [(0, 0, 1), (1, 2, 4), (0, 5, 5), (2, 6, 6)]
    assert plot_signals.label_segments(np.array([], dtype=int)) == []


def test_select_window_centers_on_first_segment() -> None:
    t = np.arange(10000, dtype=float)  # 10 с, 1 мс на рядок
    labels = np.zeros(10000, dtype=int)
    labels[4000:6000] = 3  # центр сегмента — 4999,5 мс
    labels[8000:8500] = 3  # другий сегмент ігнорується
    i0, i1 = plot_signals.select_window(t, labels, 3, duration_ms=3000)
    assert t[i0] == pytest.approx(3500, abs=1)
    assert t[i1 - 1] - t[i0] == pytest.approx(3000, abs=1)
    assert np.all(labels[i0:i1][[0, -1]] == 0)  # обидві межі сегмента потрапляють у вікно


def test_select_window_clips_to_record() -> None:
    t = np.arange(5000, dtype=float)
    labels = np.zeros(5000, dtype=int)
    labels[100:300] = 2
    i0, i1 = plot_signals.select_window(t, labels, 2, duration_ms=3000)
    assert i0 == 0 and t[i1 - 1] == pytest.approx(3000, abs=1)


def test_select_window_missing_class() -> None:
    with pytest.raises(ValueError):
        plot_signals.select_window(np.arange(10.0), np.zeros(10, dtype=int), 7)
