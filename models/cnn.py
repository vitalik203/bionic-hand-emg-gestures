"""Проста 1D-CNN для класифікації жестів за вікном sEMG.

Файл залежить лише від ``torch``. Щоб використати навчену модель поза проєктом,
достатньо скопіювати цей файл і чекпоінт::

    from cnn import load_cnn, predict_window
    model, meta = load_cnn("best_cnn.pt")
    label = predict_window(model, meta, window)   # window: [час, канали], відфільтроване

Вхід мережі — ``[N, канали, час]`` (формат Conv1d у PyTorch), вихід — логіти ``[N, класи]``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import torch
from torch import nn

PathLike = Union[str, Path]


@dataclass(frozen=True)
class CNNConfig:
    """Параметри архітектури (зберігаються в чекпоінті).

    Attributes:
        n_channels: кількість ЕМГ-каналів на вході.
        n_classes: кількість класів на виході.
        win_len: довжина вікна у відліках.
        conv_channels: кількість фільтрів у кожному згортковому блоці (2–3 блоки).
        kernel_size: ширина ядра згортки у відліках (непарна).
        dropout: частка нейронів, що вимикаються під час навчання перед повнозв'язним шаром.
        pooling: ``"flatten"`` — розгортання карт ознак (модель прив'язана до ``win_len``);
            ``"gap"`` — global average pooling, середнє по часу для кожного фільтра
            (та сама модель приймає вікна будь-якої довжини ≥ 2^кількість_блоків).
    """

    n_channels: int = 8
    n_classes: int = 6
    win_len: int = 40
    conv_channels: Tuple[int, ...] = (32, 64, 64)
    kernel_size: int = 5
    dropout: float = 0.3
    pooling: str = "flatten"

    def flat_len(self) -> int:
        """Довжина по часу після всіх MaxPool (кожен ділить на 2 з округленням вниз)."""
        n = self.win_len
        for _ in self.conv_channels:
            n //= 2
        return n


class EMGCNN(nn.Module):
    """[Conv1d → ReLU → MaxPool1d] × (2–3) → (Flatten | GlobalAvgPool) → Dropout → Linear."""

    def __init__(self, cfg: CNNConfig = CNNConfig()) -> None:
        """Args:
            cfg: параметри архітектури.

        Raises:
            ValueError: якщо блоків не 2–3, ядро парне або вікно закоротке для всіх MaxPool.
        """
        super().__init__()
        if not 2 <= len(cfg.conv_channels) <= 3:
            raise ValueError("потрібно 2–3 згорткові блоки")
        if cfg.kernel_size % 2 == 0:
            raise ValueError("kernel_size має бути непарним (щоб padding зберігав довжину)")
        if cfg.pooling not in ("flatten", "gap"):
            raise ValueError(f"невідомий pooling: {cfg.pooling}")
        if cfg.flat_len() < 1:
            raise ValueError(f"вікно {cfg.win_len} закоротке для {len(cfg.conv_channels)} MaxPool")
        self.cfg = cfg
        layers = []
        in_ch = cfg.n_channels
        for out_ch in cfg.conv_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=cfg.kernel_size, padding=cfg.kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        if cfg.pooling == "gap":
            head = [nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(cfg.dropout), nn.Linear(in_ch, cfg.n_classes)]
        else:
            head = [nn.Flatten(), nn.Dropout(cfg.dropout), nn.Linear(in_ch * cfg.flat_len(), cfg.n_classes)]
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``[N, канали, час]``.

        Returns:
            Логіти ``[N, класи]``.
        """
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    """Кількість параметрів, що навчаються."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: PathLike,
    model: EMGCNN,
    mean: Sequence[float],
    std: Sequence[float],
    index_to_label: Dict[int, int],
    extra: Dict[str, Any] | None = None,
) -> None:
    """Зберігає ваги, архітектуру, нормалізацію та відповідність класів.

    Усе зберігається як тензори або прості типи Python, тож файл
    завантажується через ``torch.load(..., weights_only=True)``.

    Args:
        path: шлях до ``.pt``.
        model: навчена модель.
        mean, std: статистики нормалізації по каналу (з train).
        index_to_label: індекс виходу мережі → мітка класу датасету.
        extra: додаткові метадані (fs, параметри вікна, метрики тощо). Перед
            збереженням проходять через JSON (``default=str``), тож будь-які
            значення стають простими типами, а ключі словників — рядками.
    """
    cfg = asdict(model.cfg)
    cfg["conv_channels"] = list(cfg["conv_channels"])
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": cfg,
        "norm_mean": torch.tensor(list(mean), dtype=torch.float32),
        "norm_std": torch.tensor(list(std), dtype=torch.float32),
        "index_to_label": {int(k): int(v) for k, v in index_to_label.items()},
        "extra": json.loads(json.dumps(extra or {}, default=str)),
    }, path)


def load_cnn(path: PathLike, device: Union[str, torch.device] = "cpu") -> Tuple[EMGCNN, Dict[str, Any]]:
    """Відновлює модель із чекпоінта, збереженого :func:`save_checkpoint`.

    Returns:
        ``(модель у режимі eval, метадані)``, де метадані містять ``norm_mean``,
        ``norm_std``, ``index_to_label``, ``config`` і ``extra``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = dict(ckpt["config"])
    cfg["conv_channels"] = tuple(cfg["conv_channels"])
    model = EMGCNN(CNNConfig(**cfg)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta


@torch.no_grad()
def predict_window(model: EMGCNN, meta: Dict[str, Any], window: Any) -> int:
    """Клас датасету для одного відфільтрованого вікна ``[час, канали]``.

    Нормалізація з чекпоінта застосовується всередині.
    """
    device = next(model.parameters()).device
    x = torch.as_tensor(window, dtype=torch.float32, device=device)
    x = (x - meta["norm_mean"].to(device)) / meta["norm_std"].to(device)
    logits = model(x.T.unsqueeze(0))
    return meta["index_to_label"][int(logits.argmax(dim=1).item())]
