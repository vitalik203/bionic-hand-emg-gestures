"""Три експерименти з SVM і 1D-CNN (контекст і правила — у CLAUDE.md).

1. **SVM проти CNN** на трьох розбиттях піддослідних 24/6/6 (зерна 0, 1, 2):
   accuracy і macro-F1, середнє ± std. Файл ``exp1.csv``.
2. **Свій проти чужого** для 6 test-піддослідних основного розбиття
   (``results/split.json``). «Свій»: навчання на серії 1 піддослідного; «чужий»:
   навчання на train-піддослідних. Тест в обох випадках — та сама серія 2
   цього піддослідного. Файл ``exp2.csv`` і графік по піддослідних.
3. **Розмір вікна** 150/200/300 мс (30/40/60 відліків при 200 Гц), крок 50 мс,
   CNN із global average pooling. Accuracy, macro-F1, сумарна затримка
   (вікно + інференс). Файл ``exp3.csv`` і графік.

Гіперпараметри не підбираються на test:

* SVM бере C і gamma з ``results/baseline.csv`` (підібрані на val основного
  розбиття). ``--retune-svm`` у експерименті 1 натомість підбирає їх на val
  кожного розбиття;
* CNN має фіксовану архітектуру, а кількість епох обирається ранньою зупинкою
  за val macro-F1. Для «свого» CNN val — друге повторення жестів у серії 1.

Кожен запуск дописує рядки в ``results/experiment_log.csv``.

Запуск::

    python experiments/run_experiments.py --estimate-only   # лише оцінка часу
    python experiments/run_experiments.py --quick           # швидка перевірка, results/quick/
    python experiments/run_experiments.py                   # усі три експерименти
    python experiments/run_experiments.py --experiments 2 3
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emg_data import BOUNDARY_LABEL, DEFAULT_DATA_DIR, load_subject_series, make_windows  # noqa: E402
from emg_preprocess import Normalizer, preprocess_segments  # noqa: E402
from experiments.baseline import FeatureConfig, get_features, model_grids, select_on_val, split_subjects  # noqa: E402
from experiments.train_cnn import (  # noqa: E402
    TrainConfig, build_datasets, evaluate, label_mapping, load_split, set_seed, time_inference_cpu,
    to_tensor_dataset, train_model,
)
from models.cnn import CNNConfig, EMGCNN, count_parameters, load_cnn  # noqa: E402

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOG_PATH = RESULTS_DIR / "experiment_log.csv"
CNN_CHECKPOINT = ROOT / "models" / "best_cnn.pt"
LOG_COLUMNS = ["timestamp", "experiment", "models", "seed", "quick", "settings", "results"]


@dataclass(frozen=True)
class RunSettings:
    """Параметри запуску; ``quick_mode`` — зменшена версія для перевірки."""

    quick: bool = False
    max_epochs: int = 50
    patience: int = 8
    seeds: Tuple[int, ...] = (0, 1, 2)
    split_sizes: Tuple[int, int, int] = (24, 6, 6)
    n_exp2_subjects: int = 6
    windows_ms: Tuple[float, ...] = (150.0, 200.0, 300.0)
    own_batch_size: int = 32
    retune_svm: bool = False

    @classmethod
    def quick_mode(cls, retune_svm: bool = False) -> "RunSettings":
        """3 епохи, 12 піддослідних (8/2/2), 2 піддослідні в експерименті 2."""
        return cls(quick=True, max_epochs=3, patience=2, split_sizes=(8, 2, 2), n_exp2_subjects=2,
                   retune_svm=retune_svm)

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / "quick" if self.quick else RESULTS_DIR

    @property
    def figures_dir(self) -> Path:
        return FIGURES_DIR / "quick" if self.quick else FIGURES_DIR


# ---------------------------------------------------------------- допоміжне

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Пише список словників у CSV (стовпці — з першого рядка)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def append_log(path: Path, experiment: str, models: Sequence[str], seed: Any, quick: bool,
               settings: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Дописує один рядок у журнал експериментів (створює файл із заголовком, якщо треба)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "experiment": experiment, "models": "+".join(models), "seed": seed, "quick": quick,
            "settings": json.dumps(settings, ensure_ascii=False, default=str),
            "results": json.dumps(results, ensure_ascii=False, default=str),
        })


def load_svm_params(path: Path = RESULTS_DIR / "baseline.csv") -> Dict[str, Any]:
    """C і gamma SVM, підібрані на val у baseline."""
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == "SVM_RBF":
                return json.loads(row["best_params"])
    raise KeyError(f"немає рядка SVM_RBF у {path}")


def make_svm(params: Dict[str, Any]):
    """SVC з RBF-ядром і заданими параметрами."""
    from sklearn.svm import SVC
    return SVC(kernel="rbf", cache_size=1000, **params)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """``(accuracy, macro-F1)``."""
    from sklearn.metrics import accuracy_score, f1_score
    return float(accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, average="macro"))


def subset_split(split: Dict[str, List[int]], sizes: Tuple[int, int, int]) -> Dict[str, List[int]]:
    """Перші ``sizes`` піддослідних кожної частини (для ``--quick``)."""
    return {k: list(split[k][:n]) for k, n in zip(("train", "val", "test"), sizes)}


def windows_with_repetition(
    filtered: np.ndarray, labels: np.ndarray, fs: float, win_ms: float, step_ms: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вікна, мітки та номер повторення жесту (1, 2, …) у межах серії.

    Повторення — порядковий номер суцільного сегмента цього класу в серії.
    Мітка кодується як ``клас * 10 + повторення``, тож ``make_windows``
    відкидає ті самі вікна, що й зі звичайними мітками.
    """
    rep = np.zeros_like(labels)
    starts = np.flatnonzero(np.diff(np.concatenate([[labels[0] - 1], labels])))
    ends = np.append(starts[1:], len(labels))
    count: Dict[int, int] = {}
    for s, e in zip(starts, ends):
        c = int(labels[s])
        if c > 0:
            count[c] = count.get(c, 0) + 1
            rep[s:e] = count[c]
    combined = np.where(labels > 0, labels * 10 + rep, labels)
    Xw, cw = make_windows(filtered, combined, fs, win_ms, step_ms, ignore_labels=(0,))
    return Xw, cw // 10, cw % 10


def train_fixed_epochs(model: EMGCNN, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int,
                       lr: float, seed: int, device: torch.device) -> EMGCNN:
    """Навчає рівно ``epochs`` епох без валідації (для донавчання на train+val)."""
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(to_tensor_dataset(X, y), batch_size=batch_size, shuffle=True, generator=gen)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def predict_cnn(model: EMGCNN, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Індекси класів для вікон ``[N, час, канали]`` (уже нормалізованих)."""
    _, _, pred, _ = evaluate(model, DataLoader(to_tensor_dataset(X, np.zeros(len(X), dtype=np.int64)),
                                               batch_size=1024), device)
    return pred


def fit_cnn(data: Dict[str, Tuple[np.ndarray, np.ndarray]], cfg: TrainConfig, arch: CNNConfig,
            device: torch.device) -> Tuple[EMGCNN, int, float]:
    """Навчає CNN із ранньою зупинкою на val; повертає ``(модель, найкраща епоха, val macro-F1)``."""
    set_seed(cfg.seed)
    model = EMGCNN(arch)
    state, history, best_epoch = train_model(model, to_tensor_dataset(*data["train"]),
                                             to_tensor_dataset(*data["val"]), cfg, device, log=lambda s: None)
    model.load_state_dict(state)
    return model.to(device).eval(), best_epoch, history[best_epoch - 1]["val_macro_f1"]


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Середнє і вибіркове std (ddof=1; для одного значення std = 0)."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


# ---------------------------------------------------------------- експеримент 1

def experiment1(s: RunSettings, feats: Dict[str, np.ndarray], data_dir: Path, device: torch.device) -> List[Dict[str, Any]]:
    """SVM проти CNN на розбиттях із зернами ``s.seeds``."""
    ids = sorted(int(i) for i in np.unique(feats["subject_id"]))
    if s.quick:
        ids = ids[:sum(s.split_sizes)]
    X, y, sid = feats["X_features"], feats["y"], feats["subject_id"]
    base_params = load_svm_params()
    base_val = set(load_split()["val"])
    rows = []
    for seed in s.seeds:
        t0 = time.perf_counter()
        split = split_subjects(ids, *s.split_sizes, seed=seed)
        m = {k: np.isin(sid, v) for k, v in split.items()}
        norm = Normalizer().fit(X[m["train"]])
        Xtr, Xva, Xte = (norm.transform(X[m[k]]) for k in ("train", "val", "test"))
        if s.retune_svm:
            make, grid = model_grids()["SVM_RBF"]
            svm, params, _ = select_on_val(make, grid, Xtr, y[m["train"]], Xva, y[m["val"]])
        else:
            params = base_params
            svm = make_svm(params).fit(Xtr, y[m["train"]])
        svm_acc, svm_f1 = metrics(y[m["test"]], svm.predict(Xte))

        cfg = TrainConfig(max_epochs=s.max_epochs, patience=s.patience, seed=seed)
        data, _ = build_datasets(split, cfg, data_dir)
        model, best_epoch, val_f1 = fit_cnn(data, cfg, CNNConfig(win_len=cfg.win_len), device)
        cnn_acc, cnn_f1 = metrics(data["test"][1], predict_cnn(model, data["test"][0], device))

        overlap = sorted(base_val & set(split["test"]))
        common = {"seed": seed, "test_subjects": json.dumps(split["test"]),
                  "baseline_val_in_test": json.dumps(overlap)}
        rows.append({**common, "model": "SVM_RBF", "test_accuracy": svm_acc, "test_macro_f1": svm_f1,
                     "n_test": int(m["test"].sum()), "details": json.dumps(params)})
        rows.append({**common, "model": "CNN_1D", "test_accuracy": cnn_acc, "test_macro_f1": cnn_f1,
                     "n_test": len(data["test"][1]),
                     "details": json.dumps({"best_epoch": best_epoch, "val_macro_f1": round(val_f1, 4)})})
        append_log(LOG_PATH, "exp1_svm_vs_cnn", ["SVM_RBF", "CNN_1D"], seed, s.quick,
                   {"split_sizes": s.split_sizes, "svm_params": params, "svm_retuned_on_val": s.retune_svm,
                    "cnn": asdict(cfg), "test_subjects": split["test"]},
                   {"SVM_RBF": {"acc": svm_acc, "f1": svm_f1}, "CNN_1D": {"acc": cnn_acc, "f1": cnn_f1},
                    "baseline_val_in_test": overlap})
        print(f"  зерно {seed}: SVM F1 {svm_f1:.3f}, CNN F1 {cnn_f1:.3f} (епоха {best_epoch}), "
              f"{time.perf_counter() - t0:.0f} с")

    for model in ("SVM_RBF", "CNN_1D"):
        accs = [r["test_accuracy"] for r in rows if r["model"] == model and isinstance(r["seed"], int)]
        f1s = [r["test_macro_f1"] for r in rows if r["model"] == model and isinstance(r["seed"], int)]
        (am, asd), (fm, fsd) = mean_std(accs), mean_std(f1s)
        for label, a, f in (("mean", am, fm), ("std", asd, fsd)):
            rows.append({"seed": label, "test_subjects": "", "baseline_val_in_test": "", "model": model,
                         "test_accuracy": a, "test_macro_f1": f, "n_test": "", "details": ""})
    write_csv(s.results_dir / "exp1.csv", rows)
    return rows


# ---------------------------------------------------------------- експеримент 2

def get_cross_cnn(s: RunSettings, split: Dict[str, List[int]], data_dir: Path,
                  device: torch.device) -> Tuple[EMGCNN, np.ndarray, np.ndarray, str]:
    """«Чужий» CNN: чекпоінт із попереднього кроку, якщо він відповідає розбиттю, інакше навчання."""
    if not s.quick and CNN_CHECKPOINT.exists():
        model, meta = load_cnn(CNN_CHECKPOINT, device)
        ok = (meta["extra"].get("split") == split and float(meta["extra"].get("fs", 0)) == TrainConfig.fs
              and model.cfg.pooling == "flatten")  # у старих чекпоінтах ключа pooling немає
        if ok:
            return model, meta["norm_mean"].numpy(), meta["norm_std"].numpy(), f"checkpoint {CNN_CHECKPOINT.name}"
    cfg = TrainConfig(max_epochs=s.max_epochs, patience=s.patience)
    data, norm = build_datasets(split, cfg, data_dir)
    model, _, _ = fit_cnn(data, cfg, CNNConfig(win_len=cfg.win_len), device)
    return model, norm.mean_, norm.std_, "trained"


def experiment2(s: RunSettings, feats: Dict[str, np.ndarray], data_dir: Path, device: torch.device) -> List[Dict[str, Any]]:
    """«Свій» (серія 1 піддослідного) проти «чужого» (train-піддослідні); тест — серія 2."""
    split = load_split()
    if s.quick:
        split = subset_split(split, s.split_sizes)
    subjects = split["test"][:s.n_exp2_subjects]
    svm_params = load_svm_params()
    cfg = TrainConfig(max_epochs=s.max_epochs, patience=s.patience)
    to_index, _ = label_mapping(cfg.classes)
    X, y, sid, ser = feats["X_features"], feats["y"], feats["subject_id"], feats["series_id"]

    tr = np.isin(sid, split["train"])
    svm_norm = Normalizer().fit(X[tr])
    svm_cross = make_svm(svm_params).fit(svm_norm.transform(X[tr]), y[tr])
    cnn_cross, c_mean, c_std, cnn_source = get_cross_cnn(s, split, data_dir, device)
    print(f"  «чужий» CNN: {cnn_source}")

    rows = []
    for subj in subjects:
        t0 = time.perf_counter()
        s1, s2 = (sid == subj) & (ser == 1), (sid == subj) & (ser == 2)
        own_norm = Normalizer().fit(X[s1])
        svm_own = make_svm(svm_params).fit(own_norm.transform(X[s1]), y[s1])
        svm_own_m = metrics(y[s2], svm_own.predict(own_norm.transform(X[s2])))
        svm_cross_m = metrics(y[s2], svm_cross.predict(svm_norm.transform(X[s2])))

        series = [(preprocess_segments(sg, lb, cfg.fs), lb) for sg, lb in load_subject_series(subj, data_dir, cfg.fs)]
        parts = []
        for f, lb in series:
            Xw, yw, rw = windows_with_repetition(f, lb, cfg.fs, cfg.win_ms, cfg.step_ms)
            keep = np.isin(yw, cfg.classes)
            parts.append((Xw[keep], np.array([to_index[int(v)] for v in yw[keep]], dtype=np.int64), rw[keep]))
        (X1, y1, r1), (X2, y2, _) = parts

        # «свій» CNN: епохи обираються за другим повторенням серії 1, потім донавчання на всій серії 1
        n_a = Normalizer().fit(X1[r1 == 1])
        own_cfg = replace(cfg, batch_size=s.own_batch_size)
        _, best_epoch, own_val_f1 = fit_cnn(
            {"train": (n_a.transform(X1[r1 == 1]), y1[r1 == 1]), "val": (n_a.transform(X1[r1 == 2]), y1[r1 == 2])},
            own_cfg, CNNConfig(win_len=cfg.win_len), device)
        n_b = Normalizer().fit(X1)
        set_seed(cfg.seed)
        cnn_own = train_fixed_epochs(EMGCNN(CNNConfig(win_len=cfg.win_len)), n_b.transform(X1), y1, best_epoch,
                                     own_cfg.batch_size, cfg.lr, cfg.seed, device)
        cnn_own_m = metrics(y2, predict_cnn(cnn_own, n_b.transform(X2), device))
        X2_cross = ((X2 - c_mean) / c_std).astype(np.float32)
        cnn_cross_m = metrics(y2, predict_cnn(cnn_cross, X2_cross, device))

        for model, own_m, cross_m, n_tr, n_te, det in (
            ("SVM_RBF", svm_own_m, svm_cross_m, int(s1.sum()), int(s2.sum()), svm_params),
            ("CNN_1D", cnn_own_m, cnn_cross_m, len(y1), len(y2),
             {"own_best_epoch": best_epoch, "own_val_macro_f1": round(own_val_f1, 4), "cross": cnn_source}),
        ):
            rows.append({
                "subject": subj, "model": model,
                "own_accuracy": own_m[0], "own_macro_f1": own_m[1],
                "cross_accuracy": cross_m[0], "cross_macro_f1": cross_m[1],
                "delta_accuracy": cross_m[0] - own_m[0], "delta_macro_f1": cross_m[1] - own_m[1],
                "n_train_own": n_tr, "n_test": n_te, "details": json.dumps(det),
            })
        print(f"  піддослідний {subj}: SVM свій/чужий F1 {svm_own_m[1]:.3f}/{svm_cross_m[1]:.3f}, "
              f"CNN {cnn_own_m[1]:.3f}/{cnn_cross_m[1]:.3f} ({time.perf_counter() - t0:.0f} с)")

    for model in ("SVM_RBF", "CNN_1D"):
        rs = [r for r in rows if r["model"] == model and r["subject"] != "mean"]
        rows.append({"subject": "mean", "model": model,
                     **{k: float(np.mean([r[k] for r in rs])) for k in (
                         "own_accuracy", "own_macro_f1", "cross_accuracy", "cross_macro_f1",
                         "delta_accuracy", "delta_macro_f1")},
                     "n_train_own": "", "n_test": "", "details": ""})
    write_csv(s.results_dir / "exp2.csv", rows)
    plot_exp2(rows, s.figures_dir / "exp2_own_vs_cross.png")
    means = {r["model"]: {k: r[k] for k in ("own_macro_f1", "cross_macro_f1", "delta_macro_f1")}
             for r in rows if r["subject"] == "mean"}
    append_log(LOG_PATH, "exp2_own_vs_cross", ["SVM_RBF", "CNN_1D"], cfg.seed, s.quick,
               {"subjects": subjects, "own": "train series 1 (CNN: epochs by repetition 2, refit on series 1)",
                "cross": "train subjects of results/split.json", "test": "series 2", "svm_params": svm_params,
                "cnn_cross_source": cnn_source, "own_batch_size": s.own_batch_size, "max_epochs": s.max_epochs},
               means)
    return rows


def plot_exp2(rows: List[Dict[str, Any]], path: Path) -> None:
    """Macro-F1 «свого» та «чужого» по піддослідних, окремо для SVM і CNN."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, facecolor="#fcfcfb")
    for ax, model in zip(axes, ("SVM_RBF", "CNN_1D")):
        rs = [r for r in rows if r["model"] == model]
        labels = [str(r["subject"]) if r["subject"] != "mean" else "середнє" for r in rs]
        x = np.arange(len(rs))
        own = [r["own_macro_f1"] for r in rs]
        cross = [r["cross_macro_f1"] for r in rs]
        ax.set_facecolor("#fcfcfb")
        for xi, a, b in zip(x, own, cross):
            ax.plot([xi, xi], [a, b], color="#c3c2b7", linewidth=2, zorder=1)
        ax.scatter(x, own, s=64, color="#2a78d6", edgecolor="#fcfcfb", linewidth=2, zorder=2, label="свій (серія 1 → 2)")
        ax.scatter(x, cross, s=64, color="#eb6834", edgecolor="#fcfcfb", linewidth=2, zorder=2, label="чужий (train → серія 2)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.axvline(len(rs) - 1.5, color="#e4e3df", linewidth=1)
        ax.set_title(model, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_xlabel("піддослідний", color="#52514e")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("macro-F1 на серії 2", color="#52514e")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Свій проти чужого: однаковий тест (серія 2 піддослідного)", x=0.01, ha="left",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


# ---------------------------------------------------------------- експеримент 3

def experiment3(s: RunSettings, data_dir: Path, device: torch.device) -> List[Dict[str, Any]]:
    """CNN із global average pooling для вікон ``s.windows_ms`` на основному розбитті."""
    split = load_split()
    if s.quick:
        split = subset_split(split, s.split_sizes)
    rows = []
    for win_ms in s.windows_ms:
        t0 = time.perf_counter()
        cfg = TrainConfig(win_ms=win_ms, max_epochs=s.max_epochs, patience=s.patience)
        data, _ = build_datasets(split, cfg, data_dir)
        arch = CNNConfig(win_len=cfg.win_len, pooling="gap")
        model, best_epoch, val_f1 = fit_cnn(data, cfg, arch, device)
        acc, f1 = metrics(data["test"][1], predict_cnn(model, data["test"][0], device))
        infer_ms, _ = time_inference_cpu(model, data["test"][0])
        row = {"win_ms": win_ms, "win_len": cfg.win_len, "step_ms": cfg.step_ms, "model": "CNN_1D_GAP",
               "val_macro_f1": val_f1, "test_accuracy": acc, "test_macro_f1": f1,
               "infer_ms_per_window_single": infer_ms, "total_latency_ms": win_ms + infer_ms,
               "n_params": count_parameters(model), "best_epoch": best_epoch,
               "n_train": len(data["train"][1]), "n_test": len(data["test"][1])}
        rows.append(row)
        append_log(LOG_PATH, "exp3_window_size", ["CNN_1D_GAP"], cfg.seed, s.quick,
                   {"cnn": asdict(cfg), "arch": asdict(arch), "split": "results/split.json"},
                   {k: row[k] for k in ("val_macro_f1", "test_accuracy", "test_macro_f1", "total_latency_ms")})
        print(f"  вікно {win_ms:g} мс ({cfg.win_len} відліків): F1 {f1:.3f}, затримка {win_ms + infer_ms:.1f} мс, "
              f"{time.perf_counter() - t0:.0f} с")
    write_csv(s.results_dir / "exp3.csv", rows)
    plot_exp3(rows, s.figures_dir / "exp3_window_size.png")
    return rows


def plot_exp3(rows: List[Dict[str, Any]], path: Path) -> None:
    """Accuracy і macro-F1 проти розміру вікна; підписи — сумарна затримка."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    w = [r["win_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(w, [r["test_accuracy"] for r in rows], marker="o", markersize=8, linewidth=2, color="#2a78d6", label="accuracy")
    ax.plot(w, [r["test_macro_f1"] for r in rows], marker="o", markersize=8, linewidth=2, color="#eb6834", label="macro-F1")
    for r in rows:
        ax.annotate(f"затримка\n{r['total_latency_ms']:.1f} мс", (r["win_ms"], min(r["test_accuracy"], r["test_macro_f1"])),
                    xytext=(0, -30), textcoords="offset points", ha="center", fontsize=8, color="#52514e")
    lo = min(min(r["test_accuracy"], r["test_macro_f1"]) for r in rows)
    hi = max(max(r["test_accuracy"], r["test_macro_f1"]) for r in rows)
    ax.set_ylim(max(0.0, lo - 0.2), min(1.0, hi + 0.05))  # широкий діапазон, щоб не роздувати різницю в ~1 пункт
    ax.set_xticks(w)
    ax.set_xticklabels([f"{x:g} мс\n({r['win_len']} відл.)" for x, r in zip(w, rows)])
    ax.set_xlabel("розмір вікна (крок 50 мс)", color="#52514e")
    ax.set_ylabel("на test", color="#52514e")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("CNN (global average pooling): точність проти розміру вікна", loc="left", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


# ---------------------------------------------------------------- оцінка часу

def estimate_runtime(s: RunSettings, feats: Dict[str, np.ndarray], data_dir: Path, device: torch.device,
                     experiments: Sequence[int]) -> Dict[str, float]:
    """Оцінює час повного запуску за вимірами підготовки даних, однієї епохи CNN і навчання SVM.

    Типова кількість епох береться з ``results/cnn.csv`` (``epochs_run`` моделі CNN_1D),
    найгірша — ``s.max_epochs``.
    """
    split = load_split()
    if s.quick:
        split = subset_split(split, s.split_sizes)
    cfg = TrainConfig(max_epochs=1, patience=1)
    t0 = time.perf_counter()
    data, _ = build_datasets(split, cfg, data_dir)
    t_build = time.perf_counter() - t0
    set_seed(0)
    t0 = time.perf_counter()
    train_model(EMGCNN(CNNConfig(win_len=cfg.win_len)), to_tensor_dataset(*data["train"]),
                to_tensor_dataset(*data["val"]), cfg, device, log=lambda m: None)
    t_epoch = time.perf_counter() - t0
    tr = np.isin(feats["subject_id"], split["train"])
    norm = Normalizer().fit(feats["X_features"][tr])
    t0 = time.perf_counter()
    make_svm(load_svm_params()).fit(norm.transform(feats["X_features"][tr]), feats["y"][tr])
    t_svm = time.perf_counter() - t0

    typical = s.max_epochs
    cnn_csv = RESULTS_DIR / "cnn.csv"
    if cnn_csv.exists() and not s.quick:
        for row in csv.DictReader(open(cnn_csv)):
            if row["model"] == "CNN_1D":
                typical = int(json.loads(row["best_params"])["epochs_run"])

    def total(epochs: int) -> Dict[str, float]:
        out = {}
        if 1 in experiments:
            out["exp1"] = len(s.seeds) * (t_build + t_svm + epochs * t_epoch)
        if 2 in experiments:
            # Груба оцінка: «свій» CNN навчається на одній серії (≈ 1/(2·n_train) train-даних),
            # двічі (вибір епох + донавчання), з меншим пакетом (множник 4 — припущення, не вимір).
            own_epoch = t_epoch / (2 * max(len(split["train"]), 1)) * 4
            cross = 0.0 if (not s.quick and CNN_CHECKPOINT.exists()) else t_build + epochs * t_epoch
            out["exp2"] = t_svm * (1 + s.n_exp2_subjects) + cross + s.n_exp2_subjects * 2 * epochs * own_epoch
        if 3 in experiments:
            out["exp3"] = sum(t_build + epochs * t_epoch * w / 200.0 for w in s.windows_ms)
        out["total"] = sum(out.values())
        return out

    est = {"t_build_s": t_build, "t_epoch_s": t_epoch, "t_svm_s": t_svm, "typical_epochs": typical,
           "typical": total(typical), "worst": total(s.max_epochs)}
    print(f"оцінка: підготовка даних {t_build:.1f} с, епоха CNN {t_epoch:.1f} с, SVM {t_svm:.1f} с")
    for k in ("typical", "worst"):
        parts = ", ".join(f"{n} ≈ {v / 60:.1f} хв" for n, v in est[k].items())
        label = f"типово (~{typical} епох)" if k == "typical" else f"найгірше ({s.max_epochs} епох)"
        print(f"  {label}: {parts}")
    return est


# ---------------------------------------------------------------- головна функція

def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Розбирає аргументи й запускає вибрані експерименти."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="мало епох і мало піддослідних; результати в results/quick/")
    ap.add_argument("--estimate-only", action="store_true", help="лише оцінити час і вийти")
    ap.add_argument("--experiments", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    ap.add_argument("--retune-svm", action="store_true", help="експ. 1: підбирати C/gamma на val кожного розбиття")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args(argv)

    s = RunSettings.quick_mode(args.retune_svm) if args.quick else RunSettings(retune_svm=args.retune_svm)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats, _ = get_features(FeatureConfig(), args.data_dir)
    print(f"пристрій: {device}; quick={s.quick}; експерименти {args.experiments}")
    est = estimate_runtime(s, feats, args.data_dir, device, args.experiments)
    if args.estimate_only:
        return {"estimate": est}

    out: Dict[str, Any] = {"estimate": est}
    t_all = time.perf_counter()
    if 1 in args.experiments:
        print("Експеримент 1: SVM проти CNN")
        out["exp1"] = experiment1(s, feats, args.data_dir, device)
    if 2 in args.experiments:
        print("Експеримент 2: свій проти чужого")
        out["exp2"] = experiment2(s, feats, args.data_dir, device)
    if 3 in args.experiments:
        print("Експеримент 3: розмір вікна")
        out["exp3"] = experiment3(s, args.data_dir, device)
    print(f"готово за {(time.perf_counter() - t_all) / 60:.1f} хв; результати в {s.results_dir.relative_to(ROOT)}/")
    return out


if __name__ == "__main__":
    main()
