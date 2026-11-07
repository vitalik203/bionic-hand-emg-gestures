"""Тести emg_preprocess на синтетичних сигналах."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import signal as sps

from emg_data import BOUNDARY_LABEL
from emg_preprocess import (
    Normalizer,
    StreamingFilter,
    band_edges,
    causal_filter,
    design_sos,
    preprocess,
    preprocess_segments,
)


def _gain_db(sos: np.ndarray, f_hz: float, fs: float) -> float:
    _, h = sps.sosfreqz(sos, worN=[f_hz], fs=fs)
    return float(20 * np.log10(np.abs(h[0]) + 1e-300))


def _sines(fs: float, dur_s: float, comps) -> np.ndarray:
    t = np.arange(int(fs * dur_s)) / fs
    return sum(a * np.sin(2 * np.pi * f * t) for f, a in comps)


# ---------- проєктування фільтра ----------

@pytest.mark.parametrize("fs,expected", [(1000.0, (20.0, 450.0)), (2000.0, (20.0, 450.0)), (200.0, (20.0, 90.0))])
def test_band_edges(fs: float, expected) -> None:
    assert band_edges(fs) == pytest.approx(expected)


def test_band_edges_too_low_fs() -> None:
    with pytest.raises(ValueError):
        band_edges(40.0)  # 0.45 * 40 = 18 Гц < 20 Гц


def test_design_frequency_response() -> None:
    fs = 1000.0
    sos = design_sos(fs)
    assert sos.shape == (5, 6)  # 4 секції Баттерворта + 1 режекторна
    assert _gain_db(sos, 50.0, fs) < -40
    assert abs(_gain_db(sos, 100.0, fs)) < 0.5
    assert _gain_db(sos, 20.0, fs) == pytest.approx(-3.0, abs=0.3)
    assert _gain_db(sos, 450.0, fs) == pytest.approx(-3.0, abs=0.3)
    assert _gain_db(sos, 2.0, fs) < -60


def test_notch_skipped_above_nyquist() -> None:
    assert design_sos(100.0).shape == (4, 6)       # 50 Гц = fs/2, режекції немає
    assert design_sos(1000.0, notch_hz=None).shape == (4, 6)


# ---------- preprocess ----------

def test_preprocess_removes_dc_low_freq_and_mains() -> None:
    fs = 1000.0
    keep = _sines(fs, 4.0, [(120.0, 0.5)])
    x = 3.0 + _sines(fs, 4.0, [(2.0, 1.0), (50.0, 1.0)]) + keep
    X = np.stack([x, -x], axis=1)
    y = preprocess(X, fs)
    # Режекторний фільтр Q=30 має смугу ~1,7 Гц і стала часу ~0,2 с: на 0,5 с від краю
    # перехідний процес ще помітний, тому перевіряємо середину 1,5–2,5 с.
    mid = slice(1500, 2500)
    np.testing.assert_allclose(y[mid, 0], keep[mid], atol=0.02)
    np.testing.assert_allclose(y[mid, 1], -keep[mid], atol=0.02)


def test_preprocess_is_zero_phase() -> None:
    fs = 1000.0
    x = _sines(fs, 2.0, [(100.0, 1.0)])[:, None]
    y = preprocess(x, fs)
    lags = np.arange(-5, 6)
    xc = [np.dot(x[500 + l:1500 + l, 0], y[500:1500, 0]) for l in lags]
    assert lags[int(np.argmax(xc))] == 0


def test_preprocess_keeps_shape_and_float_dtype() -> None:
    x = np.random.default_rng(0).normal(size=(500, 8)).astype(np.float32)
    y = preprocess(x, 1000.0)
    assert y.shape == x.shape and y.dtype == np.float32


def test_preprocess_validation() -> None:
    with pytest.raises(ValueError):
        preprocess(np.zeros(100), 1000.0)
    with pytest.raises(ValueError):
        preprocess(np.zeros((5, 2)), 1000.0)  # закороткий для filtfilt


def test_preprocess_segments_filters_each_run_separately() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(300, 2)), rng.normal(size=(400, 2))
    signals = np.concatenate([a, np.full((1, 2), 99.0), b])
    labels = np.concatenate([np.ones(300), [BOUNDARY_LABEL], 2 * np.ones(400)]).astype(int)
    y = preprocess_segments(signals, labels, 1000.0)
    np.testing.assert_allclose(y[:300], preprocess(a, 1000.0))
    np.testing.assert_array_equal(y[300], [0.0, 0.0])
    np.testing.assert_allclose(y[301:], preprocess(b, 1000.0))


# ---------- потоковий фільтр ----------

def test_streaming_matches_offline_causal() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2000, 8))
    sf = StreamingFilter(1000.0, 8)
    sizes = [1, 7, 50, 123, 0, 200, 619]
    sizes.append(len(x) - sum(sizes))
    out, pos = [], 0
    for s in sizes:
        out.append(sf.process(x[pos:pos + s]))
        pos += s
    np.testing.assert_allclose(np.concatenate(out), causal_filter(x, 1000.0), rtol=0, atol=1e-12)


def test_streaming_reset_restarts() -> None:
    x = np.random.default_rng(3).normal(size=(300, 2))
    sf = StreamingFilter(1000.0, 2)
    first = sf.process(x)
    sf.reset()
    np.testing.assert_array_equal(sf.process(x), first)


def test_streaming_wrong_channels() -> None:
    with pytest.raises(ValueError):
        StreamingFilter(1000.0, 8).process(np.zeros((10, 4)))


def test_streaming_passes_band_after_transient() -> None:
    fs = 1000.0
    x = _sines(fs, 2.0, [(120.0, 1.0), (50.0, 1.0)])[:, None]
    y = causal_filter(x, fs)
    tail = y[1000:, 0]
    assert np.sqrt(np.mean(tail ** 2)) == pytest.approx(1 / np.sqrt(2), rel=0.05)


# ---------- Normalizer ----------

def test_normalizer_fit_on_train_only() -> None:
    rng = np.random.default_rng(4)
    train = rng.normal(loc=[1.0, -2.0], scale=[2.0, 0.5], size=(5000, 2))
    test = rng.normal(loc=[3.0, 0.0], scale=[1.0, 1.0], size=(5000, 2))
    norm = Normalizer().fit(train)
    z_train = norm.transform(train)
    np.testing.assert_allclose(z_train.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z_train.std(axis=0), 1.0, atol=1e-12)
    z_test = norm.transform(test)
    np.testing.assert_allclose(z_test.mean(axis=0), (test.mean(0) - norm.mean_) / norm.std_)
    assert abs(z_test.mean(axis=0)[0]) > 0.5  # статистики test не підглядаються


def test_normalizer_accepts_windows_and_keeps_dtype() -> None:
    X = np.random.default_rng(5).normal(size=(10, 20, 3)).astype(np.float32)
    norm = Normalizer().fit(X)
    assert norm.mean_.shape == (3,)
    Z = norm.transform(X)
    assert Z.shape == X.shape and Z.dtype == np.float32


def test_normalizer_constant_channel() -> None:
    X = np.column_stack([np.ones(100), np.arange(100.0)])
    Z = Normalizer().fit_transform(X)
    assert np.all(np.isfinite(Z)) and np.all(Z[:, 0] == 0)


def test_normalizer_save_load(tmp_path: Path) -> None:
    norm = Normalizer().fit(np.random.default_rng(6).normal(size=(100, 4)))
    path = tmp_path / "norm.npz"
    norm.save(path)
    loaded = Normalizer.load(path)
    np.testing.assert_array_equal(loaded.mean_, norm.mean_)
    np.testing.assert_array_equal(loaded.std_, norm.std_)


def test_normalizer_errors(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        Normalizer().transform(np.zeros((3, 2)))
    with pytest.raises(RuntimeError):
        Normalizer().save(tmp_path / "x.npz")
    with pytest.raises(ValueError):
        Normalizer().fit(np.zeros((5, 2))).transform(np.zeros((5, 3)))
