"""Тести GestureRecognizer і make_mock_predictions на синтетичному чекпоінті та даних."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gesture_recognizer import DEFAULT_MODEL_PATH, GESTURE_NAMES, GestureRecognizer, make_mock_predictions
from models.cnn import CNNConfig, EMGCNN, save_checkpoint

HEADER = "time\t" + "\t".join(f"channel{i}" for i in range(1, 9)) + "\tclass"


@pytest.fixture()
def model_path(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    path = tmp_path / "cnn.pt"
    save_checkpoint(path, EMGCNN(CNNConfig(win_len=40)), [0.0] * 8, [1e-4] * 8,
                    {i: i + 1 for i in range(6)}, extra={"fs": 200.0, "step_ms": 50.0})
    return path


@pytest.fixture()
def rec(model_path: Path) -> GestureRecognizer:
    return GestureRecognizer(model_path)


def _window(seed: int = 0) -> np.ndarray:
    return (1e-4 * np.random.default_rng(seed).normal(size=(40, 8))).astype(np.float32)


def test_predict_keys_types_and_ranges(rec: GestureRecognizer) -> None:
    out = rec.predict(_window())
    assert set(out) == {"gesture_id", "gesture_name", "confidence", "probabilities"}
    assert out["gesture_id"] in range(1, 7) and isinstance(out["gesture_id"], int)
    assert out["gesture_name"] == GESTURE_NAMES[out["gesture_id"]]
    assert set(out["probabilities"]) == set(range(1, 7))
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["confidence"] == max(out["probabilities"].values())
    assert out["probabilities"][out["gesture_id"]] == out["confidence"]
    json.dumps(out, ensure_ascii=False)  # повідомлення серіалізується в JSON


@pytest.mark.parametrize("seed", range(5))
def test_probabilities_sum_to_one(rec: GestureRecognizer, seed: int) -> None:
    assert sum(rec.predict(_window(seed))["probabilities"].values()) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("shape", [(8, 40), (40,), (39, 8), (40, 7), (1, 40, 8)])
def test_wrong_shape_raises_clear_error(rec: GestureRecognizer, shape) -> None:
    with pytest.raises(ValueError, match=r"\[час, канали\]"):
        rec.predict(np.zeros(shape, dtype=np.float32))


def test_transposed_window_hint(rec: GestureRecognizer) -> None:
    with pytest.raises(ValueError, match="переставлені"):
        rec.predict(np.zeros((8, 40)))


def test_non_finite_raises(rec: GestureRecognizer) -> None:
    w = _window()
    w[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        rec.predict(w)


def test_eval_mode_no_grad_and_deterministic(rec: GestureRecognizer) -> None:
    assert not rec.model.training
    assert all(not p.requires_grad for p in rec.model.parameters())
    w = _window(1)
    assert rec.predict(w) == rec.predict(w)  # Dropout вимкнений


def test_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        GestureRecognizer(tmp_path / "nope.pt")


def _write_uci(path: Path, n: int, label_runs, seed: int) -> None:
    rng = np.random.default_rng(seed)
    labels = np.concatenate([[lab] * length for lab, length in label_runs])[:n]
    lines = [HEADER]
    for i in range(n):
        vals = "\t".join(f"{v:g}" for v in np.round(rng.normal(scale=3e-4, size=8), 5))
        lines.append(f"{i + 1}\t{vals}\t{labels[i]}")
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))


def test_mock_predictions_csv_format(rec: GestureRecognizer, tmp_path: Path) -> None:
    data = tmp_path / "data"
    for sid in (5, 7):
        d = data / f"{sid:02d}"
        d.mkdir(parents=True)
        runs = [(0, 500), (1, 1500), (0, 500), (2, 1500), (0, 1000)]  # 5 с при 1 кГц
        _write_uci(d / "1_raw_data_a.txt", 5000, runs, seed=sid)
        _write_uci(d / "2_raw_data_b.txt", 5000, runs, seed=sid + 100)
    out = tmp_path / "mock_predictions.csv"
    meta = make_mock_predictions(rec, out, series=1, data_dir=data, subjects=[5, 7])

    with open(out, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header == ["timestamp", "gesture_id", "confidence"]
    ts = np.array([float(r[0]) for r in rows])
    assert len(rows) == meta["n_rows"] == sum(s["n_windows"] for s in meta["segments"])
    assert ts[0] == pytest.approx(0.2)                         # перше вікно 200 мс готове через 0,2 с
    assert np.all(np.diff(ts) > 0)                             # час потоку зростає й між піддослідними
    np.testing.assert_allclose(np.diff(ts[: meta["segments"][0]["n_windows"]]), 0.05, atol=1e-9)
    assert {int(r[1]) for r in rows} <= set(range(1, 7))
    assert all(0.0 <= float(r[2]) <= 1.0 for r in rows)
    assert meta["segments"][1]["t_start"] == pytest.approx(meta["segments"][0]["t_end"])
    assert (tmp_path / "mock_predictions_meta.json").exists()


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="немає models/best_cnn.pt")
def test_real_checkpoint_loads() -> None:
    rec = GestureRecognizer()
    assert rec.input_shape == (40, 8) and rec.fs == 200.0
    assert sorted(rec.index_to_label.values()) == [1, 2, 3, 4, 5, 6]
