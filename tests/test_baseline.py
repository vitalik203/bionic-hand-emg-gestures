"""Тести допоміжних функцій experiments/baseline.py (без навчання моделей)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "baseline", Path(__file__).resolve().parents[1] / "experiments" / "baseline.py"
)
baseline = importlib.util.module_from_spec(_spec)
sys.modules["baseline"] = baseline  # потрібно для dataclass
_spec.loader.exec_module(baseline)


def test_split_disjoint_sizes_and_deterministic() -> None:
    ids = list(range(1, 37))
    s1 = baseline.split_subjects(ids, seed=42)
    s2 = baseline.split_subjects(ids, seed=42)
    assert s1 == s2
    tr, va, te = map(set, (s1["train"], s1["val"], s1["test"]))
    assert (len(tr), len(va), len(te)) == (24, 6, 6)
    assert not (tr & va or tr & te or va & te)
    assert tr | va | te == set(ids)
    assert baseline.split_subjects(ids, seed=1) != s1


def test_split_wrong_sizes() -> None:
    with pytest.raises(ValueError):
        baseline.split_subjects(range(10))


def test_locked_split_blocks_until_unlock() -> None:
    store = baseline.LockedSplit(np.zeros((3, 2)), np.zeros(3))
    with pytest.raises(RuntimeError):
        store.get()
    assert store.access_count == 0
    store.unlock()
    X, y = store.get()
    assert X.shape == (3, 2) and store.access_count == 1


def test_cache_key_changes_with_config_and_data() -> None:
    cfg = baseline.FeatureConfig()
    fp = [("01/1_raw_data_a.txt", 100, 1)]
    k = baseline.cache_key(cfg, fp)
    assert k == baseline.cache_key(cfg, list(fp))
    assert k != baseline.cache_key(replace(cfg, win_ms=250.0), fp)
    assert k != baseline.cache_key(replace(cfg, zc_ssc_threshold=0.01), fp)
    assert k != baseline.cache_key(cfg, [("01/1_raw_data_a.txt", 101, 1)])


def test_cache_roundtrip_and_stale_key(tmp_path: Path) -> None:
    data = {
        "X_features": np.arange(20.0).reshape(2, 10),
        "y": np.array([1, 2]),
        "subject_id": np.array([1, 2]),
        "series_id": np.array([1, 1]),
    }
    path = tmp_path / "features.npz"
    baseline.save_cache(path, data, "key-a", baseline.FeatureConfig())
    loaded = baseline.load_cache(path, "key-a")
    for k, v in data.items():
        np.testing.assert_array_equal(loaded[k], v)
    assert baseline.load_cache(path, "key-b") is None
    assert baseline.load_cache(tmp_path / "missing.npz", "key-a") is None


def test_select_on_val_picks_best_by_val() -> None:
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(0)
    Xtr = np.concatenate([rng.normal(0, 1, (50, 2)), rng.normal(4, 1, (50, 2))])
    ytr = np.repeat([1, 2], 50)
    Xva = np.concatenate([rng.normal(0, 1, (20, 2)), rng.normal(4, 1, (20, 2))])
    yva = np.repeat([1, 2], 20)
    _, params, log = baseline.select_on_val(
        KNeighborsClassifier, [{"n_neighbors": 1}, {"n_neighbors": 15}], Xtr, ytr, Xva, yva
    )
    assert len(log) == 2
    best = max(log, key=lambda r: r["val_macro_f1"])
    assert params == {"n_neighbors": 1 if best is log[0] else 15}
