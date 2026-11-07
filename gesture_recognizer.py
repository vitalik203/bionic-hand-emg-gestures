"""Розпізнавання жесту за одним вікном sEMG за допомогою навченої 1D-CNN.

Використання::

    from gesture_recognizer import GestureRecognizer
    rec = GestureRecognizer()                 # models/best_cnn.pt
    result = rec.predict(window)              # window: [40, 8], відфільтроване, ненормалізоване
    # {"gesture_id": 2, "gesture_name": "кулак", "confidence": 0.91, "probabilities": {1: 0.01, ...}}

``make_mock_predictions`` проганяє тестових піддослідних через каузальний фільтр і
модель так, як це відбувалося б у потоці, і пише ``mock_predictions.csv``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from emg_data import DEFAULT_DATA_DIR, load_subject_series
from emg_stream import CausalWindowFilter
from models.cnn import load_cnn

ROOT = Path(__file__).resolve().parent
PathLike = Union[str, Path]

DEFAULT_MODEL_PATH = ROOT / "models" / "best_cnn.pt"
DEFAULT_SPLIT_PATH = ROOT / "results" / "split.json"
DEFAULT_MOCK_PATH = ROOT / "results" / "mock_predictions.csv"

#: Назви жестів для повідомлень (номери — класи датасету UCI).
GESTURE_NAMES: Dict[int, str] = {
    1: "спокій",
    2: "кулак",
    3: "згинання зап'ястя",
    4: "розгинання зап'ястя",
    5: "радіальне відхилення",
    6: "ульнарне відхилення",
}


class GestureRecognizer:
    """Обгортка над чекпоінтом CNN: нормалізація, інференс без градієнтів, назви жестів."""

    def __init__(self, model_path: PathLike = DEFAULT_MODEL_PATH, device: Union[str, torch.device] = "cpu") -> None:
        """Завантажує модель, нормалізацію, відповідність класів і параметри архітектури.

        Args:
            model_path: шлях до чекпоінта, збереженого ``models.cnn.save_checkpoint``.
            device: ``"cpu"`` або ``"cuda"``.

        Raises:
            FileNotFoundError: якщо чекпоінта немає.
            ValueError: якщо класи моделі не збігаються з ``GESTURE_NAMES``.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"немає чекпоінта {path}; спершу запустіть experiments/train_cnn.py")
        self.device = torch.device(device)
        self.model, self.meta = load_cnn(path, self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.meta["config"]
        self.win_len: int = int(cfg["win_len"])
        self.n_channels: int = int(cfg["n_channels"])
        extra = self.meta.get("extra", {})
        self.fs: float = float(extra.get("fs", 200.0))
        self.step_ms: float = float(extra.get("step_ms", 50.0))
        self.index_to_label: Dict[int, int] = {int(k): int(v) for k, v in self.meta["index_to_label"].items()}
        unknown = set(self.index_to_label.values()) - set(GESTURE_NAMES)
        if unknown:
            raise ValueError(f"модель має класи без назв: {sorted(unknown)}")
        self._mean = self.meta["norm_mean"].to(self.device).float()
        self._std = self.meta["norm_std"].to(self.device).float()

    @property
    def input_shape(self) -> tuple:
        """Очікувана форма вікна ``(час, канали)``."""
        return (self.win_len, self.n_channels)

    def _check(self, window: Any) -> np.ndarray:
        x = np.asarray(window, dtype=np.float32)
        if x.shape != self.input_shape:
            hint = ""
            if x.shape == self.input_shape[::-1]:
                hint = " Схоже, осі переставлені: потрібно [час, канали], а не [канали, час]."
            raise ValueError(
                f"window має форму {self.input_shape} ([час, канали]: {self.win_len} відліків × "
                f"{self.n_channels} каналів при {self.fs:g} Гц), отримано {x.shape}.{hint}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("window містить NaN або нескінченні значення")
        return x

    def predict(self, window: Any) -> Dict[str, Any]:
        """Розпізнає жест в одному вікні.

        Args:
            window: ``[40, 8]`` — 200 мс сигналу при 200 Гц, уже відфільтрований
                (смуговий 20–90 Гц + режекторний 50 Гц), не нормалізований.

        Returns:
            ``{"gesture_id": int 1–6, "gesture_name": str, "confidence": float,
            "probabilities": {gesture_id: float}}``; ймовірності в сумі дають 1,
            ``confidence`` — ймовірність обраного жесту.

        Raises:
            ValueError: при неправильній формі або нескінченних значеннях.
        """
        x = torch.from_numpy(self._check(window)).to(self.device)
        with torch.inference_mode():
            logits = self.model(((x - self._mean) / self._std).T.unsqueeze(0))
            probs = torch.softmax(logits.double(), dim=1)[0].cpu().numpy()  # float64: сума = 1 з точністю ~1e-16
        idx = int(np.argmax(probs))
        gesture_id = self.index_to_label[idx]
        return {
            "gesture_id": gesture_id,
            "gesture_name": GESTURE_NAMES[gesture_id],
            "confidence": float(probs[idx]),
            "probabilities": {self.index_to_label[i]: float(p) for i, p in enumerate(probs)},
        }


def make_mock_predictions(
    recognizer: Optional[GestureRecognizer] = None,
    out_path: PathLike = DEFAULT_MOCK_PATH,
    split_path: PathLike = DEFAULT_SPLIT_PATH,
    series: int = 1,
    data_dir: PathLike = DEFAULT_DATA_DIR,
    subjects: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Пише ``mock_predictions.csv`` для тестових піддослідних, як у потоці.

    Серія ``series`` кожного test-піддослідного з ``split.json`` програється по
    черзі як один потік. Вікна йдуть із кроком ``step_ms`` моделі (50 мс) і
    фільтруються каузально (:class:`emg_stream.CausalWindowFilter`); між
    піддослідними стан фільтра скидається. Прогноз робиться для кожного вікна,
    включно з паузами між жестами (класом 0 датасету).

    Колонки CSV: ``timestamp`` — секунди від початку потоку, коли надійшов
    останній відлік вікна; ``gesture_id``; ``confidence``. Поруч пишеться
    ``<ім'я>_meta.json`` із порядком піддослідних і їхніми інтервалами часу.

    Returns:
        Метадані (вони ж у JSON): піддослідні, інтервали, кількість рядків і
        accuracy на вікнах із розміткою 1–6 без зміни жесту.
    """
    rec = recognizer or GestureRecognizer()
    if subjects is None:
        subjects = json.loads(Path(split_path).read_text())["test"]
    win, step = rec.win_len, int(round(rec.step_ms * rec.fs / 1000.0))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    offset, rows, segments = 0.0, [], []
    correct = labeled = 0
    for sid in subjects:
        signals, labels = load_subject_series(int(sid), data_dir, rec.fs)[series - 1]
        cwf = CausalWindowFilter(rec.fs, rec.n_channels, win, step)
        n = (len(signals) - win) // step + 1
        for k in range(n):
            start, end = k * step, k * step + win
            res = rec.predict(cwf.process(signals[start:end]))
            rows.append((round(offset + end / rec.fs, 3), res["gesture_id"], round(res["confidence"], 4)))
            lab = labels[start:end]
            if lab[0] in GESTURE_NAMES and np.all(lab == lab[0]):
                labeled += 1
                correct += int(res["gesture_id"] == lab[0])
        duration = len(signals) / rec.fs
        segments.append({"subject": int(sid), "series": series, "t_start": round(offset, 3),
                         "t_end": round(offset + duration, 3), "n_windows": n})
        offset += duration

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "gesture_id", "confidence"])
        w.writerows(rows)
    meta = {
        "csv": out_path.name, "fs": rec.fs, "win_len": win, "step_samples": step,
        "filter": "causal (emg_stream.CausalWindowFilter), reset between subjects",
        "segments": segments, "n_rows": len(rows),
        "labeled_windows": labeled, "accuracy_on_labeled_windows": correct / labeled if labeled else None,
    }
    out_path.with_name(out_path.stem + "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


if __name__ == "__main__":
    info = make_mock_predictions()
    print(json.dumps({k: v for k, v in info.items() if k != "segments"}, ensure_ascii=False, indent=2))
    for seg in info["segments"]:
        print(seg)
