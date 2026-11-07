"""Графіки сирих сигналів UCI EMG для одного піддослідного та розподіл класів.

Що будується:

* для кожного класу 1–6 — окремий рисунок з 8 каналами, 3 с навколо першого
  розміченого сегмента цього класу в серії 1. Вісь часу — реальні мітки стовпця
  ``time`` (без передискретизації), розмічена ділянка зафарбована;
* стовпчикова діаграма кількості рядків по класах 0–7 для піддослідного й
  для всіх 36 разом.

Запуск::

    python scripts/plot_signals.py --subject 1

PNG зберігаються в ``figures/``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emg_data import DEFAULT_DATA_DIR, read_raw_file, subject_files  # noqa: E402

CLASS_NAMES: Dict[int, str] = {
    0: "без розмітки",
    1: "спокій",
    2: "кулак",
    3: "згинання зап'ястя",
    4: "розгинання зап'ястя",
    5: "радіальне відведення",
    6: "ульнарне відведення",
    7: "розкрита долоня",
}
GESTURES = (1, 2, 3, 4, 5, 6)
ALL_CLASSES = tuple(range(8))
SCALE = 1e5  # значення у файлі × 1e5 → цілі від −128 до 127

# Палітра (світла тема): поверхня, тексти, серія, нейтральна заливка.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
MUTED = "#a3a29b"
SHADE = "#d8e6f7"


def label_segments(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """Розбиває масив міток на суцільні сегменти.

    Args:
        labels: мітки ``[T]``.

    Returns:
        Список ``(мітка, перший_індекс, останній_індекс)`` у порядку запису.
    """
    if len(labels) == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change - 1, [len(labels) - 1]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def select_window(
    time_ms: np.ndarray, labels: np.ndarray, cls: int, duration_ms: float = 3000.0
) -> Tuple[int, int]:
    """Знаходить ``duration_ms`` запису з центром у першому сегменті класу ``cls``.

    Вікно зсувається, якщо виходить за межі запису.

    Args:
        time_ms: час рядків у мс, зростає.
        labels: мітки рядків.
        cls: клас жесту.
        duration_ms: тривалість вікна, мс.

    Returns:
        ``(i0, i1)`` — індекси рядків, зріз ``[i0:i1]``.

    Raises:
        ValueError: якщо класу ``cls`` у записі немає.
    """
    segs = [s for s in label_segments(labels) if s[0] == cls]
    if not segs:
        raise ValueError(f"клас {cls} відсутній у записі")
    _, s, e = segs[0]
    center = 0.5 * (time_ms[s] + time_ms[e])
    t0 = center - duration_ms / 2
    t0 = min(max(t0, time_ms[0]), max(time_ms[-1] - duration_ms, time_ms[0]))
    i0 = int(np.searchsorted(time_ms, t0, side="left"))
    i1 = int(np.searchsorted(time_ms, t0 + duration_ms, side="right"))
    return i0, i1


def _style_axes(ax) -> None:
    """Приглушені осі й сітка."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_gesture(
    time_ms: np.ndarray,
    signals: np.ndarray,
    labels: np.ndarray,
    cls: int,
    window: Tuple[int, int],
    ylim: float,
    subject: int,
    series_name: str,
    out_path: Path,
) -> None:
    """Малює 8 каналів одного вікна із зафарбованою ділянкою класу ``cls``."""
    import matplotlib.pyplot as plt

    i0, i1 = window
    t = time_ms[i0:i1] / 1000.0
    lab = labels[i0:i1]
    rows_per_s = (len(t) - 1) / (t[-1] - t[0])

    fig, axes = plt.subplots(8, 1, figsize=(10, 9), sharex=True, facecolor=SURFACE)
    for ch, ax in enumerate(axes):
        ax.set_facecolor(SURFACE)
        for lab_val, s, e in label_segments(lab):
            if lab_val == cls:
                ax.axvspan(t[s], t[e], color=SHADE, linewidth=0, zorder=0)
            elif lab_val != 0:
                ax.axvspan(t[s], t[e], color=GRID, linewidth=0, zorder=0)
        ax.plot(t, signals[i0:i1, ch] * SCALE, color=SERIES, linewidth=0.7)
        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel(f"к{ch + 1}", rotation=0, ha="right", va="center", color=TEXT_PRIMARY, fontsize=9)
        _style_axes(ax)
    axes[-1].set_xlabel("час від початку запису, с (за стовпцем time)", color=TEXT_SECONDARY, fontsize=9)
    fig.suptitle(
        f"Піддослідний {subject:02d}, клас {cls} — {CLASS_NAMES[cls]}",
        color=TEXT_PRIMARY, fontsize=12, x=0.01, ha="left",
    )
    fig.text(
        0.01, 0.94,
        f"{series_name}; {t[0]:.2f}–{t[-1]:.2f} с, {len(t)} рядків (≈{rows_per_s:.0f} рядків/с).\n"
        f"Синя заливка — рядки з міткою {cls}, решта — клас 0. "
        f"Значення ×1e5 (одиниці в README не вказано).",
        color=TEXT_SECONDARY, fontsize=8, va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def count_classes(subject: int, data_dir: Path) -> Counter:
    """Рахує рядки по класах в усіх серіях піддослідного (сирі файли)."""
    c: Counter = Counter()
    for path in subject_files(subject, data_dir):
        _, _, labels = read_raw_file(path)
        c.update(labels.tolist())
    return c


def plot_class_counts(subject_counts: Counter, all_counts: Counter, subject: int, n_subjects: int, out_path: Path) -> None:
    """Дві стовпчикові діаграми: піддослідний і всі піддослідні разом."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), facecolor=SURFACE)
    panels = [
        (axes[0], subject_counts, f"Піддослідний {subject:02d}"),
        (axes[1], all_counts, f"Усі {n_subjects} піддослідних"),
    ]
    for ax, counts, title in panels:
        ax.set_facecolor(SURFACE)
        values = [counts.get(k, 0) for k in ALL_CLASSES]
        colors = [MUTED if k == 0 else SERIES for k in ALL_CLASSES]
        bars = ax.bar(ALL_CLASSES, values, color=colors, width=0.7)
        for b, v in zip(bars, values):
            ax.annotate(f"{v:,}".replace(",", " "), (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=TEXT_SECONDARY)
        ax.set_xticks(ALL_CLASSES)
        ax.set_xticklabels([f"{k}\n" + CLASS_NAMES[k].replace(" ", "\n") for k in ALL_CLASSES], fontsize=7)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
        ax.set_ylabel("рядків у файлах", color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylim(0, max(values) * 1.12)
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
        _style_axes(ax)
    fig.suptitle("Кількість рядків по класах (сирі файли, обидві серії)", color=TEXT_PRIMARY,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main(argv: List[str] | None = None) -> List[Path]:
    """Будує всі рисунки; повертає список створених файлів."""
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", type=int, default=1, help="номер піддослідного (1–36)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    path = subject_files(args.subject, args.data_dir)[0]
    time_ms, signals, labels = read_raw_file(path)
    time_ms = time_ms - time_ms[0]
    windows = {cls: select_window(time_ms, labels, cls) for cls in GESTURES}
    ylim = 1.05 * max(np.abs(signals[i0:i1]).max() for i0, i1 in windows.values()) * SCALE

    created: List[Path] = []
    for cls, win in windows.items():
        out = args.out_dir / f"subject{args.subject:02d}_class{cls}_signals.png"
        plot_gesture(time_ms, signals, labels, cls, win, ylim, args.subject, f"серія 1 ({path.name})", out)
        created.append(out)
        print(f"клас {cls}: {time_ms[win[0]] / 1000:.2f}–{time_ms[win[1] - 1] / 1000:.2f} с -> {out.relative_to(ROOT)}")

    subjects = sorted(int(p.name) for p in Path(args.data_dir).iterdir() if p.is_dir() and p.name.isdigit())
    all_counts: Counter = Counter()
    subj_counts: Counter = Counter()
    for sid in subjects:
        c = count_classes(sid, args.data_dir)
        all_counts.update(c)
        if sid == args.subject:
            subj_counts = c
    out = args.out_dir / f"class_counts_subject{args.subject:02d}_vs_all.png"
    plot_class_counts(subj_counts, all_counts, args.subject, len(subjects), out)
    created.append(out)
    print(f"класи піддослідного {args.subject:02d}: {dict(sorted(subj_counts.items()))}")
    print(f"класи всіх {len(subjects)}: {dict(sorted(all_counts.items()))}")
    return created


if __name__ == "__main__":
    main()
