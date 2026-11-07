"""Збирає таблиці результатів із ``results/*.csv`` у ``docs/results_tables.md``.

Числа не вводяться вручну: таблиці генеруються з CSV, тож після нового запуску
експериментів достатньо перезапустити скрипт::

    python scripts/build_results_md.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "results_tables.md"


def read(name: str) -> Optional[List[Dict[str, str]]]:
    """Рядки CSV або ``None``, якщо файлу немає."""
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(v: str, digits: int = 3) -> str:
    """Число з комою як десятковим роздільником; порожнє лишається порожнім."""
    if v in ("", None):
        return ""
    return f"{float(v):.{digits}f}".replace(".", ",")


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Markdown-таблиця."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def section_models() -> str:
    rows = (read("baseline.csv") or []) + (read("cnn.csv") or [])
    if not rows:
        return ""
    body = [[r["model"], num(r["test_accuracy"]), num(r["test_macro_f1"]), num(r["val_macro_f1"]),
             num(r["infer_ms_per_window_single"], 3), r["n_test"], r["best_params"].replace("|", "/")] for r in rows]
    return ("## Основне розбиття (зерно 42): baseline і CNN\n\nДжерело: `results/baseline.csv`, `results/cnn.csv`. "
            "Test — 6 піддослідних із `results/split.json`.\n\n" +
            table(["модель", "test accuracy", "test macro-F1", "val macro-F1", "мс/вікно (одне, CPU)", "n_test",
                   "параметри"], body))


def section_exp1() -> str:
    rows = read("exp1.csv")
    if not rows:
        return ""
    body = [[r["seed"], r["model"], num(r["test_accuracy"]), num(r["test_macro_f1"]), r["n_test"],
             r.get("baseline_val_in_test", "")] for r in rows]
    return ("## Експеримент 1: SVM проти CNN на трьох розбиттях 24/6/6\n\nДжерело: `results/exp1.csv`. "
            "Рядки `mean`/`std` — середнє і вибіркове std (ddof=1) по трьох розбиттях. "
            "`baseline_val_in_test` — val-піддослідні основного розбиття (на них підбирали C/gamma SVM), "
            "що потрапили в test цього розбиття.\n\n" +
            table(["зерно", "модель", "accuracy", "macro-F1", "n_test", "baseline_val_in_test"], body))


def section_exp2() -> str:
    rows = read("exp2.csv")
    if not rows:
        return ""
    body = [[r["subject"], r["model"], num(r["own_macro_f1"]), num(r["cross_macro_f1"]), num(r["delta_macro_f1"]),
             num(r["own_accuracy"]), num(r["cross_accuracy"]), r["n_train_own"], r["n_test"]] for r in rows]
    return ("## Експеримент 2: свій проти чужого (тест — серія 2 піддослідного)\n\nДжерело: `results/exp2.csv`. "
            "Δ = чужий − свій.\n\n" +
            table(["піддослідний", "модель", "свій F1", "чужий F1", "Δ F1", "свій acc", "чужий acc",
                   "n_train (свій)", "n_test"], body))


def section_exp3() -> str:
    rows = read("exp3.csv")
    if not rows:
        return ""
    body = [[num(r["win_ms"], 0), r["win_len"], num(r["val_macro_f1"]), num(r["test_accuracy"]),
             num(r["test_macro_f1"]), num(r["infer_ms_per_window_single"], 3), num(r["total_latency_ms"], 1),
             r["best_epoch"], r["n_test"]] for r in rows]
    return ("## Експеримент 3: розмір вікна (CNN із global average pooling)\n\nДжерело: `results/exp3.csv`. "
            "Сумарна затримка = довжина вікна + час інференсу одного вікна на CPU.\n\n" +
            table(["вікно, мс", "відліків", "val F1", "test acc", "test F1", "інференс, мс", "затримка, мс",
                   "епоха", "n_test"], body))


def section_filter() -> str:
    rows = read("causal_vs_filtfilt.csv")
    if not rows:
        return ""
    keys = ["rel_rmse", "corr", "lag_ms", "feat_MAV_med_rel", "feat_RMS_med_rel", "feat_WL_med_rel",
            "feat_ZC_med_rel", "feat_SSC_med_rel"]
    vals = {k: sorted(float(r[k]) for r in rows) for k in keys}
    med = lambda v: v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
    body = [[k, num(str(med(v))), num(str(v[0])), num(str(v[-1]))] for k, v in vals.items()]
    return ("## Каузальний фільтр проти нуль-фазового (filtfilt)\n\nДжерело: `results/causal_vs_filtfilt.csv` "
            f"({len(rows)} піддослідних, серія 1, сітка 1 кГц).\n\n" + table(["метрика", "медіана", "мін", "макс"], body))


def section_mock() -> str:
    path = RESULTS / "mock_predictions_meta.json"
    if not path.exists():
        return ""
    meta = json.loads(path.read_text())
    body = [[s["subject"], s["series"], num(str(s["t_start"]), 3), num(str(s["t_end"]), 3), s["n_windows"]]
            for s in meta["segments"]]
    acc = meta.get("accuracy_on_labeled_windows")
    return ("## Потоковий прогноз (mock_predictions.csv)\n\nДжерело: `results/mock_predictions_meta.json`. "
            f"Рядків: {meta['n_rows']}; accuracy на {meta['labeled_windows']} розмічених вікнах (класи 1–6 без зміни "
            f"жесту): {num(str(acc)) if acc is not None else '—'}.\n\n" +
            table(["піддослідний", "серія", "t_start, с", "t_end, с", "вікон"], body))


def main() -> Path:
    """Пише ``docs/results_tables.md``."""
    parts = [f"# Таблиці результатів\n\nЗгенеровано `scripts/build_results_md.py` "
             f"{dt.datetime.now().isoformat(timespec='minutes')} з `results/*.csv`. Не редагуйте вручну."]
    for fn in (section_models, section_exp1, section_exp2, section_exp3, section_filter, section_mock):
        s = fn()
        if s:
            parts.append(s)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n\n".join(parts) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}")
    return OUT


if __name__ == "__main__":
    main()
