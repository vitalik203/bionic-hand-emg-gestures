"""Тести EMGStreamEmulator і CausalWindowFilter."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from emg_preprocess import causal_filter
from emg_stream import CausalWindowFilter, EMGStreamEmulator


class FakeClock:
    """Віртуальний годинник: sleep лише пересуває час."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


def _record(n: int = 400, c: int = 8) -> np.ndarray:
    return np.arange(n * c, dtype=np.float32).reshape(n, c)


def test_windows_content_and_stream_time() -> None:
    clock = FakeClock()
    emu = EMGStreamEmulator(_record(), fs=200.0, clock=clock, sleep=clock.sleep)
    ws = list(emu.windows())
    assert (emu.win, emu.step) == (40, 10)
    assert len(ws) == emu.n_windows == (400 - 40) // 10 + 1
    np.testing.assert_array_equal(ws[3].data, _record()[30:70])
    assert [w.t_stream for w in ws[:3]] == pytest.approx([0.2, 0.25, 0.3])
    # віддача в реальному часі: вікно k не раніше за t_stream
    assert all(w.emitted_at >= w.t_stream - 1e-9 for w in ws)
    assert emu.is_stopped


def test_speed_factor() -> None:
    clock = FakeClock()
    emu = EMGStreamEmulator(_record(), fs=200.0, speed=10, clock=clock, sleep=clock.sleep)
    ws = list(emu.windows())
    assert ws[-1].emitted_at == pytest.approx(ws[-1].t_stream / 10, abs=1e-6)


def test_stop_ends_generator() -> None:
    clock = FakeClock()
    emu = EMGStreamEmulator(_record(), fs=200.0, clock=clock, sleep=clock.sleep)
    got = []
    for w in emu.windows():
        got.append(w)
        if w.index == 4:
            emu.stop()
    assert len(got) == 5


def test_pause_shifts_stream_time() -> None:
    clock = FakeClock()
    emu = EMGStreamEmulator(_record(), fs=200.0, clock=clock, sleep=clock.sleep)
    gen = emu.windows()
    first = next(gen)
    emu.pause()
    clock.t += 5.0          # 5 с паузи
    emu.resume()
    second = next(gen)
    assert second.emitted_at - first.emitted_at == pytest.approx(5.0 + 0.05, abs=1e-6)


def test_queue_mode_real_time() -> None:
    emu = EMGStreamEmulator(_record(200), fs=200.0, speed=20)  # 1 с запису → ~0,05 с
    t0 = time.monotonic()
    q = emu.start()
    items = []
    while (w := q.get(timeout=2)) is not None:
        items.append(w)
    emu.join(1)
    elapsed = time.monotonic() - t0
    assert len(items) == emu.n_windows == 17
    assert elapsed >= items[-1].t_stream / 20 - 0.01
    with pytest.raises(RuntimeError):
        emu.start()


def test_queue_mode_pause_and_stop() -> None:
    emu = EMGStreamEmulator(_record(400), fs=200.0, speed=20)
    q = emu.start()
    q.get(timeout=2)
    emu.pause()
    assert emu.is_paused
    time.sleep(0.1)
    emu.stop()
    rest = []
    while (w := q.get(timeout=2)) is not None:
        rest.append(w)
    assert len(rest) <= 2  # після паузи й зупинки нових вікон майже немає
    emu.join(1)


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        EMGStreamEmulator(np.zeros(100), fs=200.0)
    with pytest.raises(ValueError):
        EMGStreamEmulator(np.zeros((10, 8)), fs=200.0)
    with pytest.raises(ValueError):
        EMGStreamEmulator(_record(), fs=200.0, speed=0)


def test_causal_window_filter_matches_full_causal_filter() -> None:
    x = np.random.default_rng(0).normal(size=(1000, 8))
    cwf = CausalWindowFilter(200.0, 8, win=40, step=10)
    ref = causal_filter(x, 200.0)
    for k in range((1000 - 40) // 10 + 1):
        s = k * 10
        np.testing.assert_allclose(cwf.process(x[s:s + 40]), ref[s:s + 40], rtol=0, atol=1e-12)


def test_causal_window_filter_rejects_non_consecutive() -> None:
    x = np.random.default_rng(1).normal(size=(200, 8))
    cwf = CausalWindowFilter(200.0, 8, win=40, step=10)
    cwf.process(x[0:40])
    with pytest.raises(ValueError, match="не продовжує"):
        cwf.process(x[20:60])
    cwf.reset()
    cwf.process(x[20:60])  # після reset — нове перше вікно


def test_causal_window_filter_wrong_shape() -> None:
    with pytest.raises(ValueError):
        CausalWindowFilter(200.0, 8, 40, 10).process(np.zeros((40, 7)))
