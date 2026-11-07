"""Базові класифікатори жестів (LDA, SVM-RBF, RandomForest, kNN) на ознаках Hudgins.

Пайплайн:

1. Для кожного піддослідного серії завантажуються окремо (``load_subject_series``),
   кожна фільтрується (``preprocess_segments``, нуль-фазово) і нарізається на вікна
   200/50 мс (``make_windows``, ``ignore_labels=(0,)``). Лишаються класи 1–6,
   рахуються ознаки MAV, RMS, WL, ZC, SSC. Ненормалізовані ознаки кешуються в
   ``data/cache/features.npz`` разом із ключем конфігурації та відбитком файлів даних.
2. Піддослідні діляться 24/6/6 (train/val/test) з фіксованим зерном,
   списки id пишуться в ``results/split.json``. ``Normalizer`` вчиться лише на train.
3. Гіперпараметри кожної моделі підбираються за macro-F1 на val. Test-дані до
   кінця підбору закриті в :class:`LockedSplit`: звернення до них кидає помилку.
4. На test рахуються accuracy, macro-F1, матриця помилок і час інференсу.
   Результати: ``results/baseline.csv``, ``results/baseline_val_grid.csv``,
   ``results/confusion_<model>.png``, ``results/leakage_audit.txt``.

Запуск::

    python experiments/baseline.py            # використати кеш, якщо він актуальний
    python experiments/baseline.py --rebuild  # перерахувати ознаки
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emg_data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    UCI_CLASS_NAMES,
    load_subject_series,
    make_windows,
)
from emg_features import extract_features, feature_names  # noqa: E402
from emg_preprocess import Normalizer, preprocess_segments  # noqa: E402

SEED = 42
CACHE_PATH = ROOT / "data" / "cache" / "features.npz"
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------- конфігурація

@dataclass(frozen=True)
class FeatureConfig:
    """Параметри, від яких залежать кешовані ознаки (усі входять у ключ кешу)."""

    fs: float = 1000.0
    win_ms: float = 200.0
    step_ms: float = 50.0
    max_gap_ms: float = 100.0
    zc_ssc_threshold: float = 0.0
    classes: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    filter: str = "preprocess_segments: butter4 bandpass 20-min(450,0.45fs) + notch50 Q30, sosfiltfilt"
    version: int = 1


def data_fingerprint(data_dir: Path) -> List[Tuple[str, int, int]]:
    """Список ``(відносний шлях, розмір, mtime_ns)`` усіх файлів даних, для інвалідації кешу."""
    files = sorted(p for p in Path(data_dir).glob("[0-9][0-9]/*_raw_data_*.txt"))
    return [(str(p.relative_to(data_dir)), p.stat().st_size, p.stat().st_mtime_ns) for p in files]


def cache_key(cfg: FeatureConfig, fingerprint: Sequence[Tuple[str, int, int]]) -> str:
    """SHA-256 від конфігурації та відбитка даних."""
    payload = json.dumps({"cfg": asdict(cfg), "data": list(fingerprint)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------- ознаки й кеш

def subject_features(sid: int, cfg: FeatureConfig, data_dir: Path) -> Dict[str, np.ndarray]:
    """Ознаки всіх вікон одного піддослідного; кожна серія обробляється окремо.

    Returns:
        ``{"X": [n, 40], "y": [n], "series": [n]}``.
    """
    xs, ys, ss = [], [], []
    for k, (signals, labels) in enumerate(load_subject_series(sid, data_dir, cfg.fs, cfg.max_gap_ms)):
        filtered = preprocess_segments(signals, labels, cfg.fs)
        Xw, yw = make_windows(filtered, labels, cfg.fs, cfg.win_ms, cfg.step_ms, ignore_labels=(0,))
        keep = np.isin(yw, cfg.classes)
        xs.append(extract_features(Xw[keep], threshold=cfg.zc_ssc_threshold))
        ys.append(yw[keep])
        ss.append(np.full(int(keep.sum()), k + 1, dtype=np.int64))
    return {"X": np.concatenate(xs), "y": np.concatenate(ys), "series": np.concatenate(ss)}


def build_features(cfg: FeatureConfig, data_dir: Path, subjects: Sequence[int]) -> Dict[str, np.ndarray]:
    """Ознаки для всіх піддослідних (без нормалізації)."""
    parts = {sid: subject_features(sid, cfg, data_dir) for sid in subjects}
    return {
        "X_features": np.concatenate([p["X"] for p in parts.values()]),
        "y": np.concatenate([p["y"] for p in parts.values()]),
        "subject_id": np.concatenate([np.full(len(p["y"]), sid, dtype=np.int64) for sid, p in parts.items()]),
        "series_id": np.concatenate([p["series"] for p in parts.values()]),
    }


def save_cache(path: Path, data: Dict[str, np.ndarray], key: str, cfg: FeatureConfig) -> None:
    """Зберігає ознаки, ключ і конфігурацію в ``.npz``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **data, cache_key=np.array(key), config=np.array(json.dumps(asdict(cfg))),
        feature_names=np.array(feature_names(data["X_features"].shape[1] // 5)),
    )


def load_cache(path: Path, key: str) -> Optional[Dict[str, np.ndarray]]:
    """Повертає кеш, лише якщо його ключ збігається з ``key``; інакше ``None``."""
    if not path.exists():
        return None
    with np.load(path) as z:
        if str(z["cache_key"]) != key:
            return None
        return {k: z[k] for k in ("X_features", "y", "subject_id", "series_id")}


def get_features(cfg: FeatureConfig, data_dir: Path, rebuild: bool = False) -> Tuple[Dict[str, np.ndarray], str]:
    """Ознаки з кешу або перераховані; повертає також джерело (``"cache"``/``"built"``)."""
    fp = data_fingerprint(data_dir)
    if not fp:
        raise FileNotFoundError(f"немає файлів даних у {data_dir}")
    key = cache_key(cfg, fp)
    if not rebuild:
        cached = load_cache(CACHE_PATH, key)
        if cached is not None:
            return cached, "cache"
    subjects = sorted({int(Path(f).parts[0]) for f, _, _ in fp})
    data = build_features(cfg, data_dir, subjects)
    save_cache(CACHE_PATH, data, key, cfg)
    return data, "built"


# ---------------------------------------------------------------- розбиття

def split_subjects(
    subject_ids: Sequence[int], n_train: int = 24, n_val: int = 6, n_test: int = 6, seed: int = SEED
) -> Dict[str, List[int]]:
    """Випадкове розбиття піддослідних на train/val/test без перетинів.

    Raises:
        ValueError: якщо сума розмірів не дорівнює кількості піддослідних.
    """
    ids = np.array(sorted(set(int(s) for s in subject_ids)))
    if n_train + n_val + n_test != len(ids):
        raise ValueError(f"{n_train}+{n_val}+{n_test} != {len(ids)} піддослідних")
    perm = np.random.default_rng(seed).permutation(ids)
    return {
        "train": sorted(int(s) for s in perm[:n_train]),
        "val": sorted(int(s) for s in perm[n_train:n_train + n_val]),
        "test": sorted(int(s) for s in perm[n_train + n_val:]),
    }


class LockedSplit:
    """Сховище test-даних, закрите до завершення підбору гіперпараметрів."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X, self._y = X, y
        self.locked = True
        self.access_count = 0

    def unlock(self) -> None:
        """Відкриває доступ (викликати лише після вибору всіх моделей)."""
        self.locked = False

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        """Повертає ``(X, y)``; кидає ``RuntimeError``, поки сховище закрите."""
        if self.locked:
            raise RuntimeError("test-дані закриті до завершення підбору гіперпараметрів")
        self.access_count += 1
        return self._X, self._y


# ---------------------------------------------------------------- моделі

def model_grids() -> Dict[str, Tuple[Callable[..., Any], List[Dict[str, Any]]]]:
    """Конструктори моделей і невеликі сітки гіперпараметрів."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    return {
        "LDA": (LinearDiscriminantAnalysis, [
            {"solver": "svd"},
            {"solver": "lsqr", "shrinkage": "auto"},
            {"solver": "lsqr", "shrinkage": 0.1},
            {"solver": "lsqr", "shrinkage": 0.5},
        ]),
        "SVM_RBF": (lambda **p: SVC(kernel="rbf", cache_size=1000, **p), [
            {"C": c, "gamma": g} for c in (0.1, 1.0, 10.0, 100.0) for g in ("scale", 0.01, 0.1)
        ]),
        "RandomForest": (lambda **p: RandomForestClassifier(random_state=SEED, n_jobs=-1, **p), [
            {"n_estimators": n, "max_depth": d} for n in (100, 300) for d in (None, 20)
        ]),
        "kNN": (KNeighborsClassifier, [
            {"n_neighbors": k, "weights": w} for k in (1, 5, 15, 31) for w in ("uniform", "distance")
        ]),
    }


def select_on_val(
    make: Callable[..., Any], grid: List[Dict[str, Any]],
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
) -> Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    """Навчає кожну комбінацію на train, оцінює macro-F1 на val, повертає найкращу.

    Test-дані сюди не передаються за сигнатурою.

    Returns:
        ``(навчена_модель, найкращі_параметри, журнал_усіх_комбінацій)``.
    """
    from sklearn.metrics import f1_score

    best, best_params, best_f1, log = None, None, -1.0, []
    for params in grid:
        t0 = time.perf_counter()
        model = make(**params).fit(Xtr, ytr)
        f1 = f1_score(yva, model.predict(Xva), average="macro")
        log.append({"params": json.dumps(params), "val_macro_f1": f1, "fit_s": time.perf_counter() - t0})
        if f1 > best_f1:
            best, best_params, best_f1 = model, params, f1
    return best, best_params, log


def time_inference(model: Any, X: np.ndarray, n_single: int = 200, seed: int = SEED) -> Tuple[float, float]:
    """Час передбачення: медіана для одного вікна і середнє на вікно в пакеті, мс.

    Перед вимірюванням ``n_jobs`` переводиться в 1: для одного вікна паралелізм
    лише додає накладні витрати.
    """
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
    idx = np.random.default_rng(seed).choice(len(X), size=min(n_single, len(X)), replace=False)
    for i in idx[:10]:
        model.predict(X[i:i + 1])
    single = []
    for i in idx:
        t0 = time.perf_counter()
        model.predict(X[i:i + 1])
        single.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    model.predict(X)
    batch = (time.perf_counter() - t0) / len(X)
    return 1000 * float(np.median(single)), 1000 * batch


def time_features(cfg: FeatureConfig, norm: Normalizer, n: int = 200, seed: int = SEED) -> float:
    """Медіанний час ``extract_features`` + ``Normalizer.transform`` для одного вікна, мс."""
    win = int(round(cfg.win_ms * cfg.fs / 1000))
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(1, win, 8)).astype(np.float32)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        norm.transform(extract_features(w, threshold=cfg.zc_ssc_threshold))
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def plot_confusion(cm: np.ndarray, classes: Sequence[int], title: str, path: Path) -> None:
    """Матриця помилок: колір і число — частка рядка (recall), у дужках — кількість."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    annot = np.array([[f"{share[i, j]:.0%}\n({cm[i, j]})" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    names = [f"{c} {UCI_CLASS_NAMES[c]}" for c in classes]
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="#fcfcfb")
    sns.heatmap(share, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, square=True,
                xticklabels=names, yticklabels=names, cbar_kws={"label": "частка рядка"},
                linewidths=1, linecolor="#fcfcfb", annot_kws={"fontsize": 8}, ax=ax)
    ax.set_xlabel("передбачений клас", color="#52514e")
    ax.set_ylabel("істинний клас", color="#52514e")
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    ax.tick_params(labelsize=8, colors="#52514e")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


# ---------------------------------------------------------------- аудит витоку

@dataclass
class AuditItem:
    """Один пункт аудиту."""

    stage: str
    check: str
    status: str  # "OK", "ПРОБЛЕМА", "ЗАСТЕРЕЖЕННЯ"
    detail: str


@dataclass
class AuditContext:
    """Усе, що потрібно для перевірок аудиту."""

    data: Dict[str, np.ndarray]
    split: Dict[str, List[int]]
    masks: Dict[str, np.ndarray]
    norm: Normalizer
    cfg: FeatureConfig
    data_dir: Path
    test_store: LockedSplit
    test_access_during_selection: int
    selection_val_only: bool
    items: List[AuditItem] = field(default_factory=list)


def run_audit(ctx: AuditContext) -> List[AuditItem]:
    """Програмні перевірки витоку між train/val/test на кожному етапі пайплайна."""
    d, items = ctx.data, ctx.items
    add = lambda stage, check, ok, detail, warn=False: items.append(
        AuditItem(stage, check, "OK" if ok else ("ЗАСТЕРЕЖЕННЯ" if warn else "ПРОБЛЕМА"), detail))
    tr, va, te = (set(ctx.split[k]) for k in ("train", "val", "test"))
    all_ids = set(int(s) for s in np.unique(d["subject_id"]))

    # 1. Розбиття
    add("розбиття", "піддослідні не перетинаються між train/val/test",
        not (tr & va or tr & te or va & te), f"|train∩val|={len(tr & va)}, |train∩test|={len(tr & te)}, |val∩test|={len(va & te)}")
    add("розбиття", "розбиття покриває всіх піддослідних 24/6/6",
        (tr | va | te) == all_ids and (len(tr), len(va), len(te)) == (24, 6, 6),
        f"розміри {len(tr)}/{len(va)}/{len(te)}, усього {len(all_ids)}")
    m = ctx.masks
    covered = m["train"].astype(int) + m["val"].astype(int) + m["test"].astype(int)
    add("розбиття", "кожне вікно належить рівно одній частині",
        bool(np.all(covered == 1)), f"вікон: {len(covered)}, з них не в одній частині: {int(np.sum(covered != 1))}")
    series_split = {}
    for sid, ser in set(zip(d["subject_id"].tolist(), d["series_id"].tolist())):
        part = next(k for k in ("train", "val", "test") if sid in ctx.split[k])
        series_split.setdefault(sid, set()).add(part)
    add("розбиття", "обидві серії піддослідного в одній частині",
        all(len(v) == 1 for v in series_split.values()), "серії не розносяться між частинами")

    # 2. Фільтрація та нарізка: ознаки піддослідного не залежать від інших
    probe = ctx.split["test"][0]
    fresh = subject_features(probe, ctx.cfg, ctx.data_dir)
    rows = d["subject_id"] == probe
    same = (fresh["X"].shape == d["X_features"][rows].shape
            and np.array_equal(fresh["X"], d["X_features"][rows]) and np.array_equal(fresh["y"], d["y"][rows]))
    add("фільтри/нарізка", f"ознаки піддослідного {probe} (test), пораховані окремо від інших, збігаються з кешем",
        same, "фільтр і вікна застосовуються в межах однієї серії одного піддослідного; міжсуб'єктних статистик немає")
    add("фільтри/нарізка", "вікна не перетинають межі серій",
        True, "(за кодом) make_windows викликається окремо для кожної серії, а межі пропусків усередині серії позначено -1 (вікна з ними відкидаються)")

    # 3. Ознаки: незалежні між вікнами
    rng = np.random.default_rng(SEED)
    w = rng.normal(size=(5, 200, 8))
    add("ознаки", "ознаки вікна не залежать від інших вікон у пакеті",
        np.allclose(extract_features(w)[2], extract_features(w[2:3])[0]), "extract_features рахує кожне вікно окремо")

    # 4. Нормалізація
    Xtr = d["X_features"][m["train"]]
    exp_mean, exp_std = Xtr.mean(0), Xtr.std(0)
    all_mean = d["X_features"].mean(0)
    add("нормалізація", "статистики Normalizer дорівнюють статистикам лише train",
        np.allclose(ctx.norm.mean_, exp_mean) and np.allclose(ctx.norm.std_, np.where(exp_std < ctx.norm.eps, 1, exp_std)),
        f"max|mean - mean_train| = {np.max(np.abs(ctx.norm.mean_ - exp_mean)):.2e}")
    add("нормалізація", "перевірка чутлива: статистики всіх даних відрізняються від train",
        not np.allclose(ctx.norm.mean_, all_mean),
        f"max|mean_train - mean_all| / std = {np.max(np.abs(exp_mean - all_mean) / ctx.norm.std_):.3f}")

    # 5. Підбір гіперпараметрів
    add("підбір гіперпараметрів", "test-дані не читалися до завершення підбору",
        ctx.test_access_during_selection == 0,
        f"звернень до test під час підбору: {ctx.test_access_during_selection} (сховище кидає RuntimeError, поки закрите)")
    add("підбір гіперпараметрів", "критерій вибору рахується лише на val",
        ctx.selection_val_only, "(за кодом) select_on_val отримує тільки train і val; результати на test не впливають на вибір")
    add("підбір гіперпараметрів", "SVM gamma='scale' і масштаби ознак — з train",
        True, "(за кодом) gamma='scale' рахується sklearn із X, переданого у fit (нормалізований train)")

    # 6. Кеш
    with np.load(CACHE_PATH) as z:
        keys = set(z.files)
        cached_X = z["X_features"]
    add("кеш", "у кеші ненормалізовані ознаки й немає інформації про розбиття",
        "split" not in " ".join(keys) and not np.allclose(cached_X.mean(0), 0, atol=1e-3),
        f"ключі кешу: {sorted(keys)}")
    add("кеш", "кеш інвалідується при зміні конфігурації або файлів даних",
        True, "(за кодом і в tests/test_baseline.py) ключ = SHA-256(конфігурація + шляхи, розміри й mtime файлів); при розбіжності ознаки перераховуються")

    # 7. Застереження (не витік між частинами, але оптимізм оцінки)
    add("нарізка", "test містить лише «чисті» вікна", False,
        "вікна з переходами між жестами і клас 0 відкинуто за мітками, зокрема в test. У потоці таких вікон багато, "
        "тож реальна точність, найімовірніше, буде нижчою. Це не витік між частинами.", warn=True)
    add("фільтри", "нуль-фазовий filtfilt неможливий у реальному часі", False,
        "фільтр використовує майбутні відліки тієї ж серії, тобто той самий піддослідний і ту саму частину, тож це не витік між частинами. "
        "Але у потоці ознаки будуть іншими (див. results/causal_vs_filtfilt.csv: MAV/RMS ~6 %, WL/ZC/SSC ~20 %).", warn=True)
    add("розбиття", "один і той самий учасник під різними id", False,
        "з файлів перевірити неможливо; якщо таке є, розбиття за id не захищає від витоку", warn=True)
    return items


# ---------------------------------------------------------------- головна функція

def main(argv: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Запускає весь експеримент і повертає рядки таблиці результатів."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuild", action="store_true", help="перерахувати ознаки, ігноруючи кеш")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args(argv)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    cfg = FeatureConfig()

    t0 = time.perf_counter()
    data, source = get_features(cfg, args.data_dir, rebuild=args.rebuild)
    print(f"ознаки: {data['X_features'].shape} ({source}, {time.perf_counter() - t0:.1f} с)")

    split = split_subjects(np.unique(data["subject_id"]))
    (RESULTS_DIR / "split.json").write_text(json.dumps({"seed": SEED, **split}, indent=2))
    masks = {k: np.isin(data["subject_id"], v) for k, v in split.items()}
    X, y = data["X_features"], data["y"]
    print("розбиття:", {k: f"{len(v)} піддослідних, {int(masks[k].sum())} вікон" for k, v in split.items()})

    norm = Normalizer().fit(X[masks["train"]])
    Xtr, ytr = norm.transform(X[masks["train"]]), y[masks["train"]]
    Xva, yva = norm.transform(X[masks["val"]]), y[masks["val"]]
    test_store = LockedSplit(X[masks["test"]], y[masks["test"]])

    selected, grid_rows = {}, []
    for name, (make, grid) in model_grids().items():
        t1 = time.perf_counter()
        model, params, log = select_on_val(make, grid, Xtr, ytr, Xva, yva)
        best_f1 = max(r["val_macro_f1"] for r in log)
        selected[name] = (model, params, best_f1)
        grid_rows += [{"model": name, **r} for r in log]
        print(f"{name}: найкращі {params}, val macro-F1 {best_f1:.3f} ({time.perf_counter() - t1:.1f} с)")
    access_during_selection = test_store.access_count

    test_store.unlock()
    Xte_raw, yte = test_store.get()
    Xte = norm.transform(Xte_raw)
    classes = list(cfg.classes)
    feat_ms = time_features(cfg, norm)
    rows = []
    for name, (model, params, val_f1) in selected.items():
        pred = model.predict(Xte)
        acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred, average="macro")
        cm = confusion_matrix(yte, pred, labels=classes)
        plot_confusion(cm, classes, f"{name}: test accuracy {acc:.3f}, macro-F1 {f1:.3f}",
                       RESULTS_DIR / f"confusion_{name}.png")
        single_ms, batch_ms = time_inference(model, Xte)
        rows.append({
            "model": name, "best_params": json.dumps(params), "val_macro_f1": round(val_f1, 4),
            "test_accuracy": round(acc, 4), "test_macro_f1": round(f1, 4),
            "infer_ms_per_window_single": round(single_ms, 4), "infer_ms_per_window_batch": round(batch_ms, 5),
            "features_ms_per_window": round(feat_ms, 4),
            "n_train": int(masks["train"].sum()), "n_val": int(masks["val"].sum()), "n_test": int(masks["test"].sum()),
        })

    for fname, table in (("baseline.csv", rows), ("baseline_val_grid.csv", grid_rows)):
        with open(RESULTS_DIR / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)

    items = run_audit(AuditContext(data, split, masks, norm, cfg, args.data_dir, test_store,
                                   access_during_selection, selection_val_only=True))
    lines = [f"[{it.status}] {it.stage}: {it.check}. {it.detail}" for it in items]
    (RESULTS_DIR / "leakage_audit.txt").write_text("\n".join(lines) + "\n")

    print("\nTEST (результат):")
    cols = ["model", "test_accuracy", "test_macro_f1", "val_macro_f1", "infer_ms_per_window_single",
            "infer_ms_per_window_batch", "best_params"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))
    print(f"ознаки + нормалізація для одного вікна: {feat_ms:.3f} мс")
    print("\nАУДИТ ВИТОКУ:")
    print("\n".join(lines))
    return rows


if __name__ == "__main__":
    main()
