"""Порівняння каузального потокового фільтра з нуль-фазовим (filtfilt) на реальних записах UCI.

Для кожного піддослідного береться серія 1 на сітці 1 кГц, найдовша безперервна
ділянка. Рахується:

* ``stream_vs_offline_rel_max`` — max |потік блоками по 50 мс − sosfilt на всьому
  сигналі| / RMS сигналу. Перевіряє, що стан між блоками зберігається правильно;
* ``rel_rmse`` — ||каузальний − filtfilt|| / ||filtfilt|| (без першої 1 с);
* ``corr`` — кореляція Пірсона між ними;
* ``lag_ms`` — зсув (0–30 мс), за якого кореляція максимальна (запізнення каузального);
* ``rel_rmse_aligned`` — rel_rmse після компенсації цього зсуву;
* ``feat_<F>_med_rel`` — медіана |F_каузальний − F_filtfilt| / |F_filtfilt| по вікнах
  200/50 мс класів 1–7 і каналах; для ZC/SSC додатково медіана абсолютної різниці.

Запуск::

    python scripts/compare_causal_filter.py

Результат: ``results/causal_vs_filtfilt.csv`` і зведення в stdout.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emg_data import BOUNDARY_LABEL, DEFAULT_DATA_DIR, make_windows, read_raw_file, resample_uniform, subject_files  # noqa: E402
from emg_features import FEATURE_ORDER, extract_features  # noqa: E402
from emg_preprocess import StreamingFilter, causal_filter, preprocess  # noqa: E402

FS = 1000.0
BLOCK = 50          # відліків у блоці потоку (50 мс)
SKIP = 1000         # перша 1 с не враховується (перехідний процес каузального фільтра)
MAX_LAG = 30        # мс


def longest_run(signals: np.ndarray, labels: np.ndarray):
    """Найдовша безперервна ділянка без ``BOUNDARY_LABEL``."""
    valid = (labels != BOUNDARY_LABEL).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], valid, [0]])))
    starts, stops = edges[::2], edges[1::2]
    i = int(np.argmax(stops - starts))
    return signals[starts[i]:stops[i]], labels[starts[i]:stops[i]]


def compare_subject(sid: int) -> Dict[str, float]:
    """Усі метрики для серії 1 одного піддослідного."""
    t, s, l = read_raw_file(subject_files(sid, DEFAULT_DATA_DIR)[0])
    sig, lab = resample_uniform(t, s, l, fs=FS)
    sig, lab = longest_run(sig.astype(np.float64), lab)

    zero = preprocess(sig, FS)
    offline = causal_filter(sig, FS)
    sf = StreamingFilter(FS, sig.shape[1])
    stream = np.concatenate([sf.process(sig[i:i + BLOCK]) for i in range(0, len(sig), BLOCK)])
    sig_rms = np.sqrt(np.mean(sig ** 2))

    a, b = offline[SKIP:], zero[SKIP:]
    rel = lambda u, v: float(np.linalg.norm(u - v) / np.linalg.norm(v))
    corr_at = [np.corrcoef(a[lag:].ravel(), b[:len(b) - lag].ravel())[0, 1] for lag in range(MAX_LAG + 1)]
    lag = int(np.argmax(corr_at))

    row: Dict[str, float] = {
        "subject": sid,
        "n_samples": len(sig),
        "stream_vs_offline_rel_max": float(np.max(np.abs(stream - offline)) / sig_rms),
        "rel_rmse": rel(a, b),
        "corr": float(corr_at[0]),
        "lag_ms": lag * 1000.0 / FS,
        "rel_rmse_aligned": rel(a[lag:], b[:len(b) - lag]),
    }

    lab = lab.copy()
    lab[:SKIP] = 0
    Xz, _ = make_windows(zero, lab, FS, ignore_labels=(0,))
    Xc, _ = make_windows(offline, lab, FS, ignore_labels=(0,))
    Fz = extract_features(Xz).reshape(len(Xz), -1, len(FEATURE_ORDER))
    Fc = extract_features(Xc).reshape(len(Xc), -1, len(FEATURE_ORDER))
    row["n_windows"] = len(Xz)
    for j, name in enumerate(FEATURE_ORDER):
        z, c = Fz[..., j], Fc[..., j]
        row[f"feat_{name}_med_rel"] = float(np.median(np.abs(c - z) / np.maximum(np.abs(z), 1e-12)))
        if name in ("ZC", "SSC"):
            row[f"feat_{name}_med_abs"] = float(np.median(np.abs(c - z)))
            row[f"feat_{name}_med_value"] = float(np.median(z))
    return row


def main() -> List[Dict[str, float]]:
    """Рахує метрики для всіх піддослідних і пише CSV."""
    subjects = sorted(int(p.name) for p in DEFAULT_DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    rows = [compare_subject(sid) for sid in subjects]

    out = ROOT / "results" / "causal_vs_filtfilt.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    keys = [k for k in rows[0] if k not in ("subject", "n_samples", "n_windows")]
    print(f"{'метрика':28s} {'суб.01':>10s} {'медіана':>10s} {'мін':>10s} {'макс':>10s}")
    for k in keys:
        v = np.array([r[k] for r in rows])
        print(f"{k:28s} {rows[0][k]:10.4g} {np.median(v):10.4g} {v.min():10.4g} {v.max():10.4g}")
    print(f"-> {out.relative_to(ROOT)} ({len(rows)} піддослідних)")
    return rows


if __name__ == "__main__":
    main()
