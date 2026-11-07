"""Класичні ознаки sEMG у часовій області (набір Hudgins): MAV, RMS, WL, ZC, SSC.

Вихід :func:`extract_features` має форму ``[N, канали * 5]``, порядок —
спочатку канал, потім ознака::

    ch1_MAV, ch1_RMS, ch1_WL, ch1_ZC, ch1_SSC, ch2_MAV, ...

Назви стовпців повертає :func:`feature_names`.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

#: Порядок ознак усередині блоку одного каналу.
FEATURE_ORDER: Tuple[str, ...] = ("MAV", "RMS", "WL", "ZC", "SSC")


def feature_names(n_channels: int) -> List[str]:
    """Назви стовпців у порядку виходу :func:`extract_features`.

    Args:
        n_channels: кількість каналів.

    Returns:
        Список ``["ch1_MAV", "ch1_RMS", ..., "chC_SSC"]``.
    """
    return [f"ch{c + 1}_{f}" for c in range(n_channels) for f in FEATURE_ORDER]


def extract_features(X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Обчислює MAV, RMS, WL, ZC, SSC для кожного вікна й каналу без циклів по вікнах.

    Нехай ``x_1 … x_T`` — відліки одного каналу у вікні, ``ε = threshold``:

    * MAV = mean |x_i|;
    * RMS = sqrt(mean x_i²);
    * WL  = Σ |x_{i+1} − x_i| (сума, тож залежить від довжини вікна);
    * ZC  = кількість i, де x_i · x_{i+1} < 0 і |x_i − x_{i+1}| ≥ ε
      (точні нулі перетином не вважаються);
    * SSC = кількість внутрішніх i, де (x_i − x_{i−1}) · (x_i − x_{i+1}) > 0
      і (|x_i − x_{i−1}| ≥ ε або |x_i − x_{i+1}| ≥ ε).

    Поріг ε у тих самих одиницях, що й X: якщо X нормалізовано, поріг теж має
    бути в одиницях std.

    Args:
        X: вікна ``[N, час, канали]``, час ≥ 3.
        threshold: поріг шуму ε ≥ 0 для ZC і SSC.

    Returns:
        ``[N, канали * 5]`` float64.

    Raises:
        ValueError: при неправильній формі, вікні коротшому за 3 відліки або ε < 0.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError("X має форму [N, час, канали]")
    if X.shape[1] < 3:
        raise ValueError("для SSC потрібно щонайменше 3 відліки у вікні")
    if threshold < 0:
        raise ValueError("threshold має бути ≥ 0")
    n, _, c = X.shape

    mav = np.abs(X).mean(axis=1)
    rms = np.sqrt(np.square(X).mean(axis=1))

    d = np.diff(X, axis=1)  # [N, T-1, C]
    abs_d = np.abs(d)
    wl = abs_d.sum(axis=1)
    zc = ((X[:, :-1] * X[:, 1:] < 0) & (abs_d >= threshold)).sum(axis=1)

    d_prev = d[:, :-1]   # x_i - x_{i-1}
    d_next = -d[:, 1:]   # x_i - x_{i+1}
    ssc = ((d_prev * d_next > 0) & ((abs_d[:, :-1] >= threshold) | (abs_d[:, 1:] >= threshold))).sum(axis=1)

    feats = np.stack([mav, rms, wl, zc, ssc], axis=2)  # [N, C, 5]
    return feats.reshape(n, c * len(FEATURE_ORDER))
