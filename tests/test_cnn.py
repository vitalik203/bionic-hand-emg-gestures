"""Тести models/cnn.py та циклу навчання experiments/train_cnn.py на синтетичних даних."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.train_cnn import TrainConfig, label_mapping, set_seed, to_tensor_dataset, train_model
from models.cnn import CNNConfig, EMGCNN, count_parameters, load_cnn, predict_window, save_checkpoint


@pytest.mark.parametrize("win_len", [40, 200])
def test_output_shape(win_len: int) -> None:
    model = EMGCNN(CNNConfig(win_len=win_len))
    out = model(torch.zeros(5, 8, win_len))
    assert out.shape == (5, 6)


def test_parameter_count_default() -> None:
    # conv: 8·32·5+32, 32·64·5+64, 64·64·5+64; fc: 64·(40/8)·6+6
    expected = (8 * 32 * 5 + 32) + (32 * 64 * 5 + 64) + (64 * 64 * 5 + 64) + (64 * 5 * 6 + 6)
    assert count_parameters(EMGCNN(CNNConfig(win_len=40))) == expected == 34086


@pytest.mark.parametrize(
    "kw", [{"conv_channels": (32,)}, {"conv_channels": (8, 8, 8, 8)}, {"kernel_size": 4}, {"win_len": 4}]
)
def test_invalid_config(kw) -> None:
    with pytest.raises(ValueError):
        EMGCNN(CNNConfig(**kw))


def test_unknown_monitor_rejected() -> None:
    X, y = _synthetic(5, seed=0)
    with pytest.raises(ValueError):
        train_model(EMGCNN(CNNConfig(n_classes=3)), to_tensor_dataset(X, y), to_tensor_dataset(X, y),
                    TrainConfig(monitor="acc"), torch.device("cpu"), log=lambda s: None)


def test_label_mapping() -> None:
    to_index, to_label = label_mapping((1, 2, 3, 4, 5, 6))
    assert to_index == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    assert to_label == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}


def _synthetic(n_per_class: int, seed: int, win_len: int = 40, n_classes: int = 3):
    """Клас k: синус амплітуди 3 на каналі k, решта каналів — шум 0.3."""
    rng = np.random.default_rng(seed)
    t = np.arange(win_len)
    X, y = [], []
    for k in range(n_classes):
        for _ in range(n_per_class):
            w = 0.3 * rng.normal(size=(win_len, 8))
            w[:, k] += 3 * np.sin(2 * np.pi * t / 10 + rng.uniform(0, 2 * np.pi))
            X.append(w)
            y.append(k)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def test_quick_training_learns_synthetic() -> None:
    set_seed(0)
    Xtr, ytr = _synthetic(100, seed=1)
    Xva, yva = _synthetic(30, seed=2)
    cfg = TrainConfig(batch_size=32, max_epochs=15, patience=5, seed=0)
    model = EMGCNN(CNNConfig(n_classes=3, win_len=40))
    state, history, best_epoch = train_model(
        model, to_tensor_dataset(Xtr, ytr), to_tensor_dataset(Xva, yva), cfg, torch.device("cpu"), log=lambda s: None
    )
    assert 1 <= best_epoch <= len(history) <= 15
    assert history[best_epoch - 1]["val_acc"] >= 0.9
    assert history[best_epoch - 1]["val_macro_f1"] == max(h["val_macro_f1"] for h in history)
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert set(state) == set(model.state_dict())


def test_training_is_deterministic() -> None:
    Xtr, ytr = _synthetic(20, seed=1)
    Xva, yva = _synthetic(10, seed=2)
    cfg = TrainConfig(batch_size=16, max_epochs=2, patience=5, seed=3)
    runs = []
    for _ in range(2):
        set_seed(3)
        model = EMGCNN(CNNConfig(n_classes=3))
        _, history, _ = train_model(
            model, to_tensor_dataset(Xtr, ytr), to_tensor_dataset(Xva, yva), cfg, torch.device("cpu"), log=lambda s: None
        )
        runs.append(history)
    assert runs[0] == runs[1]


def test_checkpoint_roundtrip_without_training_code(tmp_path: Path) -> None:
    set_seed(0)
    model = EMGCNN(CNNConfig(n_classes=3, win_len=40)).eval()
    mean, std = [0.5] * 8, [2.0] * 8
    path = tmp_path / "m.pt"
    # Регресія: torch.__version__ (TorchVersion) і numpy-скаляри ламали weights_only-завантаження.
    extra = {"fs": 200.0, "torch_version": torch.__version__, "val_f1": np.float64(0.5)}
    save_checkpoint(path, model, mean, std, {0: 1, 1: 2, 2: 6}, extra=extra)

    raw = torch.load(path, map_location="cpu", weights_only=True)  # без довіри до pickle
    assert raw["config"]["win_len"] == 40 and raw["extra"]["fs"] == 200.0
    assert raw["extra"]["torch_version"] == str(torch.__version__) and raw["extra"]["val_f1"] == 0.5

    loaded, meta = load_cnn(path)
    x = torch.randn(4, 8, 40)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))

    window = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32)
    with torch.no_grad():
        idx = int(model(((torch.from_numpy(window) - 0.5) / 2.0).T.unsqueeze(0)).argmax())
    assert predict_window(loaded, meta, window) == {0: 1, 1: 2, 2: 6}[idx]
