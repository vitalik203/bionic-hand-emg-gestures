"""Тести emg_features на синтетичних сигналах із відомими значеннями ознак."""

from __future__ import annotations

import numpy as np
import pytest

from emg_features import FEATURE_ORDER, extract_features, feature_names

IDX = {name: i for i, name in enumerate(FEATURE_ORDER)}


def _one(x: np.ndarray, **kw) -> dict:
    """Ознаки одного одноканального вікна у вигляді словника."""
    f = extract_features(x.reshape(1, -1, 1), **kw)[0]
    return {name: f[i] for name, i in IDX.items()}


@pytest.mark.parametrize("c", [0.7, -0.7])
def test_constant(c: float) -> None:
    f = _one(np.full(100, c))
    assert f["MAV"] == pytest.approx(abs(c))
    assert f["RMS"] == pytest.approx(abs(c))
    assert f["WL"] == 0 and f["ZC"] == 0 and f["SSC"] == 0


def test_zeros() -> None:
    f = _one(np.zeros(50))
    assert all(v == 0 for v in f.values())


def test_sine_known_values() -> None:
    a, k, t_len = 2.0, 5, 1000  # 5 повних періодів
    n = np.arange(t_len)
    x = a * np.sin(2 * np.pi * k * n / t_len + 0.3)  # зсув фази: жоден відлік не дорівнює 0
    f = _one(x)
    assert f["RMS"] == pytest.approx(a / np.sqrt(2), rel=1e-9)  # точно для цілого числа періодів
    assert f["MAV"] == pytest.approx(2 * a / np.pi, rel=1e-3)
    assert f["WL"] < 4 * a * k  # вершини між відліками й T−1 різниць дають трохи менше 4A·k
    assert f["ZC"] == 2 * k
    assert f["SSC"] == 2 * k                                     # максимум + мінімум за період


def test_sine_wavelength_exact() -> None:
    # Косинус із періодом 200: вершини точно на відліках, вікно включає кінцевий відлік,
    # тож довжина хвилі дорівнює повній варіації 4A за період.
    a, k, period = 2.0, 5, 200
    x = a * np.cos(2 * np.pi * np.arange(k * period + 1) / period)
    assert _one(x)["WL"] == pytest.approx(4 * a * k, rel=1e-12)


def test_sawtooth_known_values() -> None:
    # Період 10: −0.9, −0.7, …, 0.9; 4 періоди, 40 відліків.
    period = -1 + (2 * np.arange(10) + 1) / 10
    x = np.tile(period, 4)
    f = _one(x)
    assert f["MAV"] == pytest.approx(0.5)
    assert f["RMS"] == pytest.approx(np.sqrt(0.33))
    # 4 підйоми × 9 кроків × 0,2 + 3 спади × 1,8
    assert f["WL"] == pytest.approx(4 * 9 * 0.2 + 3 * 1.8)
    # 4 перетини на підйомах (−0.1 → 0.1) + 3 на спадах (0.9 → −0.9)
    assert f["ZC"] == 7
    # 3 внутрішні вершини (0.9) + 3 внутрішні западини (−0.9); крайні відліки не рахуються
    assert f["SSC"] == 6


def test_threshold_suppresses_small_zc_and_ssc() -> None:
    x = 0.01 * np.sin(2 * np.pi * 5 * np.arange(1000) / 1000 + 0.3)
    f0 = _one(x)
    f_hi = _one(x, threshold=0.1)
    assert f0["ZC"] == 10 and f0["SSC"] == 10
    assert f_hi["ZC"] == 0 and f_hi["SSC"] == 0
    assert f_hi["MAV"] == f0["MAV"] and f_hi["WL"] == f0["WL"]  # поріг не впливає на амплітудні ознаки


def test_threshold_ssc_uses_either_slope() -> None:
    # Вершина з різними схилами: |Δ| = 0.5 зліва і 0.05 справа.
    x = np.array([0.0, 0.5, 0.45])
    assert _one(x, threshold=0.4)["SSC"] == 1
    assert _one(x, threshold=0.6)["SSC"] == 0


def test_shape_and_order() -> None:
    n, t_len = 4, 200
    sine = np.sin(2 * np.pi * 3 * np.arange(t_len) / t_len + 0.3)
    X = np.zeros((n, t_len, 3))
    X[:, :, 0] = 0.5          # канал 1 — константа
    X[:, :, 1] = sine         # канал 2 — синус
    X[:, :, 2] = -2.0 * sine  # канал 3 — синус ×(−2)
    F = extract_features(X)
    assert F.shape == (n, 3 * 5) and F.dtype == np.float64
    names = feature_names(3)
    assert names[:6] == ["ch1_MAV", "ch1_RMS", "ch1_WL", "ch1_ZC", "ch1_SSC", "ch2_MAV"]
    col = {name: i for i, name in enumerate(names)}
    assert F[0, col["ch1_MAV"]] == pytest.approx(0.5)
    assert F[0, col["ch1_ZC"]] == 0
    assert F[0, col["ch2_ZC"]] == 6
    assert F[0, col["ch3_RMS"]] == pytest.approx(2 * F[0, col["ch2_RMS"]])
    assert F[0, col["ch3_SSC"]] == F[0, col["ch2_SSC"]] == 6
    np.testing.assert_array_equal(F, np.tile(F[0], (n, 1)))  # однакові вікна → однакові ознаки


def test_matches_per_window_reference() -> None:
    """Векторизований результат збігається з простою покроковою реалізацією."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 50, 3))
    thr = 0.3
    ref = []
    for w in X:
        row = []
        for ch in w.T:
            d = np.diff(ch)
            zc = sum(1 for i in range(len(ch) - 1) if ch[i] * ch[i + 1] < 0 and abs(d[i]) >= thr)
            ssc = sum(
                1 for i in range(1, len(ch) - 1)
                if (ch[i] - ch[i - 1]) * (ch[i] - ch[i + 1]) > 0
                and (abs(ch[i] - ch[i - 1]) >= thr or abs(ch[i] - ch[i + 1]) >= thr)
            )
            row += [np.abs(ch).mean(), np.sqrt((ch ** 2).mean()), np.abs(d).sum(), zc, ssc]
        ref.append(row)
    np.testing.assert_allclose(extract_features(X, threshold=thr), np.array(ref))


@pytest.mark.parametrize(
    "X,kw",
    [(np.zeros((10, 2)), {}), (np.zeros((1, 2, 1)), {}), (np.zeros((1, 10, 1)), {"threshold": -1.0})],
)
def test_validation(X, kw) -> None:
    with pytest.raises(ValueError):
        extract_features(X, **kw)
