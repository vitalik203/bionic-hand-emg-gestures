"""Тести допоміжних функцій experiments/run_experiments.py (без навчання на реальних даних)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import experiments.run_experiments as rx
from experiments.run_experiments import (
    RunSettings,
    get_cross_cnn,
    append_log,
    mean_std,
    subset_split,
    train_fixed_epochs,
    windows_with_repetition,
)
from emg_data import BOUNDARY_LABEL
from models.cnn import CNNConfig, EMGCNN, save_checkpoint


def test_windows_with_repetition_numbers_segments_per_class() -> None:
    # порядок як у датасеті: 1, 2, потім 1, 2 ще раз, між ними клас 0
    labels = np.array([0] * 5 + [1] * 10 + [0] * 5 + [2] * 10 + [0] * 5 + [1] * 10 + [0] * 5 + [2] * 10 + [0] * 5)
    signals = np.arange(len(labels) * 2, dtype=np.float32).reshape(-1, 2)
    X, y, rep = windows_with_repetition(signals, labels, fs=1000.0, win_ms=4, step_ms=2)
    assert set(zip(y.tolist(), rep.tolist())) == {(1, 1), (2, 1), (1, 2), (2, 2)}
    first_1 = X[(y == 1) & (rep == 1)][:, 0, 0] / 2
    second_1 = X[(y == 1) & (rep == 2)][:, 0, 0] / 2
    assert first_1.max() < 15 <= 35 <= second_1.min()  # перше повторення раніше за друге


def test_windows_with_repetition_drops_boundary() -> None:
    labels = np.array([1] * 6 + [BOUNDARY_LABEL] + [1] * 6)
    X, y, rep = windows_with_repetition(np.zeros((13, 1), np.float32), labels, 1000.0, 3, 1)
    assert len(y) == 8 and set(y.tolist()) == {1}
    assert set(rep.tolist()) == {1, 2}  # межа розриває сегмент на два


def test_subset_split() -> None:
    split = {"train": list(range(24)), "val": list(range(24, 30)), "test": list(range(30, 36))}
    assert subset_split(split, (8, 2, 2)) == {"train": list(range(8)), "val": [24, 25], "test": [30, 31]}


def test_quick_settings() -> None:
    q = RunSettings.quick_mode()
    assert q.quick and q.max_epochs < RunSettings().max_epochs and sum(q.split_sizes) == 12
    assert q.results_dir.name == "quick" and RunSettings().results_dir.name == "results"


def test_mean_std() -> None:
    assert mean_std([1.0, 2.0, 3.0]) == pytest.approx((2.0, 1.0))
    assert mean_std([5.0]) == (5.0, 0.0)


def test_append_log_creates_header_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    append_log(path, "exp1", ["SVM", "CNN"], 0, True, {"a": 1}, {"f1": 0.5})
    append_log(path, "exp3", ["CNN"], 42, False, {"win": 150}, {"f1": 0.7})
    rows = list(csv.DictReader(open(path)))
    assert [r["experiment"] for r in rows] == ["exp1", "exp3"]
    assert rows[0]["models"] == "SVM+CNN" and json.loads(rows[1]["results"]) == {"f1": 0.7}
    assert rows[0]["timestamp"] and rows[0]["quick"] == "True"


def test_train_fixed_epochs_runs_exact_epochs() -> None:
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(size=(40, 40, 8)).astype(np.float32)
    y = np.repeat([0, 1], 20)
    model = EMGCNN(CNNConfig(n_classes=2))
    before = [p.detach().clone() for p in model.parameters()]
    train_fixed_epochs(model, X, y, epochs=2, batch_size=16, lr=1e-3, seed=0, device=torch.device("cpu"))
    assert not model.training
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    unchanged = EMGCNN(CNNConfig(n_classes=2))
    init = [p.detach().clone() for p in unchanged.parameters()]
    train_fixed_epochs(unchanged, X, y, epochs=0, batch_size=16, lr=1e-3, seed=0, device=torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(init, unchanged.parameters()))


def test_get_cross_cnn_uses_old_checkpoint_without_pooling(tmp_path: Path, monkeypatch) -> None:
    """Регресія: чекпоінт до появи параметра pooling падав з KeyError."""
    split = {"train": [1, 2], "val": [3], "test": [4]}
    path = tmp_path / "best_cnn.pt"
    save_checkpoint(path, EMGCNN(CNNConfig(win_len=40)), [0.5] * 8, [2.0] * 8, {i: i + 1 for i in range(6)},
                    extra={"split": split, "fs": 200.0})
    ckpt = torch.load(path, weights_only=True)
    del ckpt["config"]["pooling"]
    torch.save(ckpt, path)
    monkeypatch.setattr(rx, "CNN_CHECKPOINT", path)
    model, mean, std, source = get_cross_cnn(RunSettings(), split, tmp_path, torch.device("cpu"))
    assert source.startswith("checkpoint") and model.cfg.pooling == "flatten"
    np.testing.assert_allclose(mean, 0.5)
    np.testing.assert_allclose(std, 2.0)
