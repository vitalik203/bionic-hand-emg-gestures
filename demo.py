"""Демонстрація розпізнавання жестів у (прискореному) реальному часі.

Запис тестового піддослідного програється через :class:`emg_stream.EMGStreamEmulator`.
Кожне вікно фільтрується каузально зі станом (:class:`emg_stream.CausalWindowFilter`:
перше вікно — повністю, далі лише нові ``step`` відліків, решта з буфера) і
розпізнається :class:`gesture_recognizer.GestureRecognizer`.

Виводиться розпізнаний жест, упевненість і затримка обробки (від моменту, коли
надійшов останній відлік вікна, до готового прогнозу). Друкуються лише зміни
жесту (і кожне ``--print-every``-те вікно, якщо задано).

Запуск::

    python demo.py --speed 10
    python demo.py --subject 9 --series 2 --speed 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from emg_data import DEFAULT_DATA_DIR, load_subject_series
from emg_preprocess import preprocess_segments
from emg_stream import CausalWindowFilter, EMGStreamEmulator
from gesture_recognizer import DEFAULT_MODEL_PATH, DEFAULT_SPLIT_PATH, GESTURE_NAMES, GestureRecognizer


def label_name(label: int) -> str:
    """Назва мітки датасету для виводу (0 — пауза без розмітки)."""
    return GESTURE_NAMES.get(int(label), "пауза/без розмітки" if label == 0 else f"клас {label}")


def main(argv: Optional[List[str]] = None) -> dict:
    """Програє запис і друкує прогнози; повертає підсумкову статистику."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", type=int, default=None, help="за замовчуванням — перший test-піддослідний")
    ap.add_argument("--series", type=int, default=1, choices=[1, 2])
    ap.add_argument("--speed", type=float, default=10.0, help="у скільки разів швидше за реальний час")
    ap.add_argument("--print-every", type=int, default=0, help="додатково друкувати кожне N-те вікно")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args(argv)

    rec = GestureRecognizer(args.model)
    subject = args.subject or json.loads(Path(DEFAULT_SPLIT_PATH).read_text())["test"][0]
    signals, labels = load_subject_series(subject, args.data_dir, rec.fs)[args.series - 1]
    emu = EMGStreamEmulator(signals, rec.fs, win_ms=rec.win_len * 1000 / rec.fs, step_ms=rec.step_ms, speed=args.speed)
    cwf = CausalWindowFilter(rec.fs, rec.n_channels, emu.win, emu.step)
    print(f"піддослідний {subject}, серія {args.series}: {len(signals) / rec.fs:.1f} с запису, "
          f"{emu.n_windows} вікон по {emu.win} відліків (крок {emu.step}), fs={rec.fs:g} Гц, speed={args.speed:g}")
    print(f"{'час, с':>7} | {'розпізнано':<22} | {'упевн.':>6} | {'затримка':>8} | мітка датасету")

    latencies, preds, t_wall = [], [], time.monotonic()
    last = None
    q = emu.start()
    while True:
        w = q.get()
        if w is None:
            break
        res = rec.predict(cwf.process(w.data))
        latency_ms = (time.monotonic() - w.emitted_at) * 1000
        latencies.append(latency_ms)
        preds.append((w.start, w.end, res["gesture_id"]))
        if res["gesture_id"] != last or (args.print_every and w.index % args.print_every == 0):
            print(f"{w.t_stream:7.2f} | {res['gesture_id']} {res['gesture_name']:<20} | {res['confidence']:6.2f} | "
                  f"{latency_ms:6.2f} мс | {label_name(labels[w.end - 1])}")
            last = res["gesture_id"]
    emu.join()
    wall = time.monotonic() - t_wall

    # точність на вікнах із розміткою 1–6 без зміни жесту; для порівняння — офлайн нуль-фазовий фільтр
    def score(pairs):
        ok = [g == labels[s] for s, e, g in pairs if labels[s] in GESTURE_NAMES and np.all(labels[s:e] == labels[s])]
        return float(np.mean(ok)) if ok else float("nan"), len(ok)

    stream_acc, n_lab = score(preds)
    offline = preprocess_segments(signals, labels, rec.fs)
    off_pairs = [(s, e, rec.predict(offline[s:e])["gesture_id"]) for s, e, _ in preds]
    offline_acc, _ = score(off_pairs)

    lat = np.array(latencies)
    window_ms = emu.win * 1000 / rec.fs
    stats = {
        "subject": subject, "series": args.series, "speed": args.speed, "windows": len(preds),
        "wall_s": wall, "expected_wall_s": len(signals) / rec.fs / args.speed,
        "latency_ms_median": float(np.median(lat)), "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_ms_max": float(lat.max()), "window_ms": window_ms,
        "accuracy_stream_causal": stream_acc, "accuracy_offline_zero_phase": offline_acc, "labeled_windows": n_lab,
    }
    print(f"\nвікон: {len(preds)}; тривалість {wall:.1f} с (очікувано {stats['expected_wall_s']:.1f} с при speed={args.speed:g})")
    print(f"затримка обробки (фільтр + модель + черга): медіана {stats['latency_ms_median']:.2f} мс, "
          f"95-й перцентиль {stats['latency_ms_p95']:.2f} мс, макс. {stats['latency_ms_max']:.2f} мс")
    print(f"повна затримка рішення ≈ вікно {window_ms:.0f} мс + обробка {stats['latency_ms_median']:.2f} мс "
          f"(без кроку 50 мс і затримки самого фільтра)")
    print(f"accuracy на {n_lab} розмічених вікнах: потік (каузальний фільтр) {stream_acc:.3f}; "
          f"офлайн (нуль-фазовий, як під час навчання) {offline_acc:.3f}")
    return stats


if __name__ == "__main__":
    main()
