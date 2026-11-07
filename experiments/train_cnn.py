"""Навчання 1D-CNN на вікнах sEMG із тим самим розбиттям піддослідних, що й baseline.

Пайплайн:

1. Піддослідні беруться з ``results/split.json``. Кожна серія завантажується на
   сітці ``fs``, фільтрується (``preprocess_segments``) і нарізається на вікна
   200/50 мс (``ignore_labels=(0,)``, лишаються класи 1–6). ``Normalizer``
   вчиться на відфільтрованих безперервних сигналах лише train-піддослідних.
   Мітки 1–6 перетворюються на індекси 0–5.
2. Модель :class:`models.cnn.EMGCNN` навчається з Adam і cross-entropy, до 50 епох,
   з ранньою зупинкою за val loss. Найкращий стан пишеться в ``models/best_cnn.pt``
   разом із нормалізацією, відповідністю класів і параметрами архітектури.
3. Test-дані закриті в ``LockedSplit`` до кінця навчання. Результати:
   ``results/cnn.csv`` (формат як у ``results/baseline.csv``),
   ``results/confusion_<model>.png``, ``figures/<model>_training_curves.png``.

Запуск::

    python experiments/train_cnn.py              # fs=200 Гц, вікно [40, 8]
    python experiments/train_cnn.py --fs 1000    # сітка як у baseline, вікно [200, 8]
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emg_data import BOUNDARY_LABEL, DEFAULT_DATA_DIR, UCI_CLASS_NAMES, load_subject_series, make_windows  # noqa: E402
from emg_preprocess import Normalizer, preprocess_segments  # noqa: E402
from experiments.baseline import LockedSplit, plot_confusion  # noqa: E402
from models.cnn import CNNConfig, EMGCNN, count_parameters, load_cnn, save_checkpoint  # noqa: E402

SPLIT_PATH = ROOT / "results" / "split.json"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"
BASELINE_COLUMNS = [
    "model", "best_params", "val_macro_f1", "test_accuracy", "test_macro_f1",
    "infer_ms_per_window_single", "infer_ms_per_window_batch", "features_ms_per_window",
    "n_train", "n_val", "n_test",
]


@dataclass(frozen=True)
class TrainConfig:
    """Параметри даних і навчання."""

    fs: float = 200.0
    win_ms: float = 200.0
    step_ms: float = 50.0
    classes: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    batch_size: int = 256
    lr: float = 1e-3
    max_epochs: int = 50
    patience: int = 8
    monitor: str = "val_macro_f1"  # або "val_loss"
    seed: int = 42

    @property
    def win_len(self) -> int:
        """Довжина вікна у відліках."""
        return int(round(self.win_ms * self.fs / 1000.0))


# ---------------------------------------------------------------- відтворюваність і пристрій

def set_seed(seed: int) -> None:
    """Фіксує зерна random, numpy і torch та вмикає детерміновані cuDNN-алгоритми."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(preferred: Optional[str] = None) -> torch.device:
    """CUDA, якщо доступна (Colab з GPU), інакше CPU. ``preferred`` перекриває вибір."""
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- дані

def label_mapping(classes: Sequence[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """``(мітка → індекс, індекс → мітка)``, наприклад 1→0 … 6→5."""
    to_index = {int(c): i for i, c in enumerate(classes)}
    return to_index, {i: c for c, i in to_index.items()}


def load_split(path: Path = SPLIT_PATH) -> Dict[str, List[int]]:
    """Списки id піддослідних train/val/test із ``split.json`` baseline."""
    raw = json.loads(Path(path).read_text())
    return {k: [int(s) for s in raw[k]] for k in ("train", "val", "test")}


def build_datasets(
    split: Dict[str, List[int]], cfg: TrainConfig, data_dir: Path = DEFAULT_DATA_DIR
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Normalizer]:
    """Вікна ``[N, час, канали]`` (float32, нормалізовані) та індекси класів для кожної частини.

    Нормалізація вчиться лише на безперервних відфільтрованих сигналах
    train-піддослідних (без службових відліків ``BOUNDARY_LABEL``).
    """
    filtered: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for sid in sorted(set(split["train"]) | set(split["val"]) | set(split["test"])):
        filtered[sid] = [
            (preprocess_segments(s, l, cfg.fs), l) for s, l in load_subject_series(sid, data_dir, cfg.fs)
        ]
    train_samples = np.concatenate(
        [f[l != BOUNDARY_LABEL] for sid in split["train"] for f, l in filtered[sid]]
    )
    norm = Normalizer().fit(train_samples)

    to_index, _ = label_mapping(cfg.classes)
    out = {}
    for part in ("train", "val", "test"):
        xs, ys = [], []
        for sid in split[part]:
            for f, l in filtered[sid]:
                Xw, yw = make_windows(f, l, cfg.fs, cfg.win_ms, cfg.step_ms, ignore_labels=(0,))
                keep = np.isin(yw, cfg.classes)
                xs.append(Xw[keep])
                ys.append(yw[keep])
        X = norm.transform(np.concatenate(xs)).astype(np.float32)
        y = np.array([to_index[int(v)] for v in np.concatenate(ys)], dtype=np.int64)
        out[part] = (X, y)
    return out, norm


def to_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    """``[N, час, канали]`` → тензори ``[N, канали, час]`` для Conv1d."""
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1))), torch.from_numpy(y))


# ---------------------------------------------------------------- навчання

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """``(середній loss, accuracy, передбачення, істинні індекси)`` у режимі eval."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, correct, n, preds, targets = 0.0, 0, 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total += loss_fn(logits, yb).item()
        p = logits.argmax(dim=1)
        correct += (p == yb).sum().item()
        n += len(yb)
        preds.append(p.cpu().numpy())
        targets.append(yb.cpu().numpy())
    return total / n, correct / n, np.concatenate(preds), np.concatenate(targets)


def train_model(
    model: nn.Module,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    cfg: TrainConfig,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, float]], int]:
    """Adam + cross-entropy з ранньою зупинкою за ``cfg.monitor`` на val.

    За замовчуванням критерій — val macro-F1 (той самий, що й для вибору
    гіперпараметрів baseline). Train loss/accuracy рахуються як середні за епоху
    під час навчання (з увімкненим Dropout); val — окремим проходом у режимі eval.

    Returns:
        ``(найкращий state_dict, історія по епохах, номер найкращої епохи)``.
    """
    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=1024)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    from sklearn.metrics import f1_score

    if cfg.monitor not in ("val_macro_f1", "val_loss"):
        raise ValueError(f"невідомий monitor: {cfg.monitor}")
    sign = 1.0 if cfg.monitor == "val_macro_f1" else -1.0  # більше — краще
    best_score, best_state, best_epoch, bad, history = -float("inf"), None, 0, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            n += len(yb)
        val_loss, val_acc, vp, vt = evaluate(model, val_loader, device)
        val_f1 = float(f1_score(vt, vp, average="macro"))
        history.append({"epoch": epoch, "train_loss": total / n, "train_acc": correct / n,
                        "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1})
        score = sign * history[-1][cfg.monitor]
        improved = score > best_score + 1e-4
        if improved:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
        log(f"епоха {epoch:2d}: train loss {total / n:.4f} acc {correct / n:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} F1 {val_f1:.3f}{' *' if improved else ''}")
        if bad >= cfg.patience:
            log(f"рання зупинка: {cfg.monitor} не покращувався {cfg.patience} епох")
            break
    return best_state, history, best_epoch


# ---------------------------------------------------------------- звітність

def plot_curves(history: List[Dict[str, float]], best_epoch: int, title: str, path: Path) -> None:
    """Loss і accuracy по епохах для train і val; пунктиром — найкраща епоха."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")
    for ax, key, ylabel in ((axes[0], "loss", "cross-entropy"), (axes[1], "acc", "accuracy")):
        ax.set_facecolor("#fcfcfb")
        ax.plot(ep, [h[f"train_{key}"] for h in history], color="#2a78d6", linewidth=2, label="train")
        ax.plot(ep, [h[f"val_{key}"] for h in history], color="#eb6834", linewidth=2, label="val")
        ax.axvline(best_epoch, color="#a3a29b", linestyle="--", linewidth=1)
        ax.text(best_epoch, ax.get_ylim()[1], f" найкраща епоха {best_epoch}", va="top", fontsize=8, color="#52514e")
        ax.set_xlabel("епоха", color="#52514e")
        ax.set_ylabel(ylabel, color="#52514e")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


@torch.no_grad()
def time_inference_cpu(model: EMGCNN, X: np.ndarray, n_single: int = 200, seed: int = 42) -> Tuple[float, float]:
    """Медіанний час одного вікна та середній час на вікно в пакеті на CPU, мс."""
    model = copy.deepcopy(model).to("cpu").eval()
    Xt = torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1)))
    idx = np.random.default_rng(seed).choice(len(Xt), size=min(n_single, len(Xt)), replace=False)
    for i in idx[:20]:
        model(Xt[i:i + 1])
    single = []
    for i in idx:
        t0 = time.perf_counter()
        model(Xt[i:i + 1]).argmax(dim=1)
        single.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    for start in range(0, len(Xt), 1024):
        model(Xt[start:start + 1024]).argmax(dim=1)
    batch = (time.perf_counter() - t0) / len(Xt)
    return 1000 * float(np.median(single)), 1000 * batch


def time_normalization(norm: Normalizer, win_len: int, n: int = 200, seed: int = 42) -> float:
    """Медіанний час ``Normalizer.transform`` для одного вікна, мс (аналог стовпця ознак baseline)."""
    w = np.random.default_rng(seed).normal(size=(win_len, len(norm.mean_))).astype(np.float32)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        norm.transform(w)
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def upsert_rows(path: Path, rows: List[Dict[str, object]], key: str = "model") -> None:
    """Записує рядки в CSV, замінюючи наявні з тим самим ``key``."""
    existing: List[Dict[str, object]] = []
    if path.exists():
        with open(path, newline="") as f:
            existing = [r for r in csv.DictReader(f)]
    names = {r[key] for r in rows}
    merged = [r for r in existing if r[key] not in names] + rows
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BASELINE_COLUMNS)
        w.writeheader()
        w.writerows(merged)


# ---------------------------------------------------------------- головна функція

def main(argv: Optional[List[str]] = None) -> Dict[str, object]:
    """Готує дані, навчає CNN, оцінює на test і зберігає результати."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fs", type=float, default=TrainConfig.fs, help="частота сітки, Гц (200 → вікно 40 відліків)")
    ap.add_argument("--epochs", type=int, default=TrainConfig.max_epochs)
    ap.add_argument("--patience", type=int, default=TrainConfig.patience)
    ap.add_argument("--monitor", default=TrainConfig.monitor, choices=["val_macro_f1", "val_loss"],
                    help="критерій ранньої зупинки й вибору найкращої епохи")
    ap.add_argument("--device", default=None, help="cpu / cuda; за замовчуванням автоматично")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args(argv)

    cfg = TrainConfig(fs=args.fs, max_epochs=args.epochs, patience=args.patience, monitor=args.monitor)
    tag = ("" if cfg.fs == TrainConfig.fs else f"_fs{int(cfg.fs)}") + ("" if cfg.monitor == TrainConfig.monitor else "_valloss")
    model_name = f"CNN_1D{tag}"
    set_seed(cfg.seed)
    device = pick_device(args.device)
    for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(exist_ok=True)

    split = load_split()
    t0 = time.perf_counter()
    data, norm = build_datasets(split, cfg, args.data_dir)
    shapes = {k: v[0].shape for k, v in data.items()}
    print(f"пристрій: {device}; fs={cfg.fs:g} Гц, вікно {cfg.win_len} відліків; дані {shapes} "
          f"({time.perf_counter() - t0:.1f} с)")
    _, index_to_label = label_mapping(cfg.classes)
    test_store = LockedSplit(*data["test"])

    arch = CNNConfig(n_channels=shapes["train"][2], n_classes=len(cfg.classes), win_len=cfg.win_len)
    model = EMGCNN(arch)
    n_params = count_parameters(model)
    print(f"параметрів: {n_params:,}".replace(",", " "))
    print(model)

    train_ds, val_ds = to_tensor_dataset(*data["train"]), to_tensor_dataset(*data["val"])
    t0 = time.perf_counter()
    best_state, history, best_epoch = train_model(model, train_ds, val_ds, cfg, device)
    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)

    val_loader = DataLoader(val_ds, batch_size=1024)
    _, val_acc, val_pred, val_true = evaluate(model, val_loader, device)
    val_f1 = f1_score(val_true, val_pred, average="macro")

    ckpt_path = MODELS_DIR / f"best_cnn{tag}.pt"
    save_checkpoint(ckpt_path, model, norm.mean_.tolist(), norm.std_.tolist(), index_to_label, extra={
        "fs": cfg.fs, "win_ms": cfg.win_ms, "step_ms": cfg.step_ms, "win_len": cfg.win_len,
        "input_layout": "[N, channels, time]; window [time, channels] -> transpose",
        "filter": "emg_preprocess.preprocess_segments (sosfiltfilt) before windowing",
        "class_names": {int(c): UCI_CLASS_NAMES[c] for c in cfg.classes},
        "best_epoch": best_epoch, "val_loss": history[best_epoch - 1]["val_loss"],
        "val_acc": val_acc, "val_macro_f1": float(val_f1), "seed": cfg.seed,
        "split": {k: list(v) for k, v in split.items()}, "torch_version": str(torch.__version__),
        "monitor": cfg.monitor,
    })
    reloaded, _ = load_cnn(ckpt_path)
    probe = val_ds.tensors[0][:64]
    with torch.no_grad():
        same = torch.allclose(reloaded(probe), model.to("cpu").eval()(probe), atol=1e-6)
    print(f"чекпоінт: {ckpt_path.relative_to(ROOT)}; перезавантажена модель дає ті самі виходи: {same}")
    model.to(device)

    access_during_training = test_store.access_count
    test_store.unlock()
    Xte, yte = test_store.get()
    _, _, te_pred, te_true = evaluate(model, DataLoader(to_tensor_dataset(Xte, yte), batch_size=1024), device)
    acc = accuracy_score(te_true, te_pred)
    f1 = f1_score(te_true, te_pred, average="macro")
    labels_true = np.array([index_to_label[int(i)] for i in te_true])
    labels_pred = np.array([index_to_label[int(i)] for i in te_pred])
    cm = confusion_matrix(labels_true, labels_pred, labels=list(cfg.classes))
    plot_confusion(cm, cfg.classes, f"{model_name}: test accuracy {acc:.3f}, macro-F1 {f1:.3f}",
                   RESULTS_DIR / f"confusion_{model_name}.png")
    plot_curves(history, best_epoch, f"{model_name}: навчання (fs={cfg.fs:g} Гц, вікно {cfg.win_len} відліків, "
                f"найкраща епоха за {cfg.monitor})",
                FIGURES_DIR / f"{model_name}_training_curves.png")
    single_ms, batch_ms = time_inference_cpu(model, Xte)
    norm_ms = time_normalization(norm, cfg.win_len)

    params = {**asdict(arch), "lr": cfg.lr, "batch_size": cfg.batch_size, "fs": cfg.fs, "monitor": cfg.monitor,
              "best_epoch": best_epoch, "epochs_run": len(history), "n_params": n_params}
    params["conv_channels"] = list(params["conv_channels"])
    row = {
        "model": model_name, "best_params": json.dumps(params), "val_macro_f1": round(float(val_f1), 4),
        "test_accuracy": round(float(acc), 4), "test_macro_f1": round(float(f1), 4),
        "infer_ms_per_window_single": round(single_ms, 4), "infer_ms_per_window_batch": round(batch_ms, 5),
        "features_ms_per_window": round(norm_ms, 4),
        "n_train": len(data["train"][1]), "n_val": len(data["val"][1]), "n_test": len(yte),
    }
    upsert_rows(RESULTS_DIR / "cnn.csv", [row])
    with open(RESULTS_DIR / f"{model_name}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    print(f"\nзвернень до test під час навчання: {access_during_training}; час навчання {train_s:.1f} с")
    print(" | ".join(BASELINE_COLUMNS[:7]))
    print(" | ".join(str(row[c]) for c in BASELINE_COLUMNS[:7]))
    return row


if __name__ == "__main__":
    main()
