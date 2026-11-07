# Біонічний протез руки: розпізнавання жестів за sEMG

Код до магістерської роботи «Біонічний протез руки з системою розпізнавання жестів
на основі нейромережі». Мета — класифікувати жести руки за поверхневою ЕМГ із
затримкою ≤ 200–300 мс і стійкістю до нових користувачів.

**Стан:** готові завантаження даних, фільтрація, нормалізація, ознаки, графіки й
базові класифікатори (LDA, SVM, RandomForest, kNN) і проста 1D-CNN. Емулятора потоку
(`emg_stream.py` поки порожній) ще немає.

## Дані

Датасет — [UCI EMG data for gestures](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures)
(id 481): 36 піддослідних, браслет Myo, 8 каналів, по 2 серії на піддослідного.
Дані в репозиторій не входять (`data/` у `.gitignore`). Завантажте й розпакуйте
архів так, щоб вийшла структура:

```
data/EMG_data_for_gestures-master/
├── README.txt
├── 01/
│   ├── 1_raw_data_13-12_22.03.16.txt
│   └── 2_raw_data_13-13_22.03.16.txt
└── … 36/
```

Інша тека передається параметром `data_dir=` у `load_subject`.

Класи: 1 спокій, 2 кулак, 3 згинання зап'ястя, 4 розгинання, 5 радіальне
відведення, 6 ульнарне відведення. Клас 0 означає дані без розмітки (64 % рядків),
його ігноруємо. Клас 7 (розкрита долоня) є лише в піддослідних 11 і 30.

### Особливості даних, перевірені на файлах

Повна інвентаризація та список неперевіреного — у [docs/datasets.md](docs/datasets.md).

- **Частота не 200 Гц.** Рядки йдуть із кроком ~1 мс (~970 рядків/с), але
  значення оновлюються лише ~100 разів/с. Між оновленнями рядки повторюються.
  Тому `load_subject` переводить запис на рівномірну сітку **1 кГц** за стовпцем
  `time`, і саме `fs=1000` треба передавати далі. У 200-мс вікні при цьому лише
  ~20 різних відліків на канал.
- У суб'єкта 34 є пропуск запису на 13,8 с і обрізаний останній рядок. Пропуски
  й межі між серіями позначаються міткою `-1`, і вікна з нею відкидаються.
- На графіках сигналів (`figures/`) межі розмітки не збігаються з видимою зміною
  активності: активність видно і в класі 0 поруч із жестом.

## Встановлення

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Код запускався на Python 3.9.6. Цільові середовища — Python 3.11+ і Google Colab;
там запуск ще не перевірявся.

## Швидкий старт

```python
import numpy as np
from emg_data import BOUNDARY_LABEL, load_subject, make_windows
from emg_preprocess import Normalizer, preprocess_segments
from emg_features import extract_features

FS = 1000.0

def load_filtered(sid):
    signals, labels = load_subject(sid, fs=FS)
    return preprocess_segments(signals, labels, FS), labels   # фільтр до нарізки

train_ids, test_ids = range(1, 29), range(29, 37)             # розбиття лише за піддослідними
train = [load_filtered(s) for s in train_ids]
norm = Normalizer().fit(np.concatenate([f[l != BOUNDARY_LABEL] for f, l in train]))

signals, labels = train[0]
X, y = make_windows(norm.transform(signals), labels, FS,
                    win_ms=200, step_ms=50, ignore_labels=(0, 7))
F = extract_features(X)   # X: (786, 200, 8), F: (786, 40) для піддослідного 1
```

## Модулі

| Файл | Що робить |
|---|---|
| [emg_data.py](emg_data.py) | `load_subject(id)` повертає сигнал `[час, 8]` і мітки на рівномірній сітці; `make_windows(...)` повертає `X [N, час, канали]` та `y [N]` і відкидає вікна, де змінюється жест |
| [emg_preprocess.py](emg_preprocess.py) | `preprocess`: смуговий фільтр 20–min(450, 0,45·fs) Гц і режекторний 50 Гц, нуль-фазово (`sosfiltfilt`). `preprocess_segments` фільтрує кожну безперервну ділянку окремо. `StreamingFilter` — каузальний фільтр зі станом між блоками для потоку. `Normalizer` — z-нормалізація по каналу з `fit`/`transform`/`save`/`load` |
| [emg_features.py](emg_features.py) | `extract_features(X, threshold)` повертає `[N, канали·5]` у порядку MAV, RMS, WL, ZC, SSC для кожного каналу; назви стовпців дає `feature_names()` |

## Скрипти

```bash
python scripts/plot_signals.py --subject 1      # PNG у figures/
python scripts/compare_causal_filter.py         # CSV у results/
```

- `plot_signals.py` будує 8 каналів по 3 с для кожного жесту 1–6 і діаграму
  кількості відліків по класах для піддослідного та для всіх 36 разом.
- `compare_causal_filter.py` порівнює потоковий каузальний фільтр із filtfilt на
  серії 1 усіх піддослідних. Медіанні результати:
  - обробка блоками дає рівно той самий результат, що й фільтр на всьому сигналі;
  - запізнення каузального фільтра ~24 мс;
  - ознаки MAV і RMS розходяться на ~6 %, а WL, ZC і SSC — на ~20 %.

  Тобто ознаки, навчені на filtfilt, у потоці будуть зміщені.

## Базові класифікатори

```bash
python experiments/baseline.py            # --rebuild, щоб перерахувати кеш ознак
```

Ознаки кешуються в `data/cache/features.npz`. Піддослідні діляться 24/6/6
(зерно 42, `results/split.json`), гіперпараметри обираються на val, test
відкривається лише наприкінці. Під час запуску виконується аудит витоку даних
(`results/leakage_audit.txt`).

Результат на test (6 піддослідних, 4 280 вікон; `results/baseline.csv`):

| Модель | accuracy | macro-F1 |
|---|---:|---:|
| LDA | 0,710 | 0,699 |
| SVM-RBF | 0,779 | 0,775 |
| RandomForest | 0,767 | 0,763 |
| kNN | 0,741 | 0,735 |

Це одне розбиття, і test містить лише «чисті» вікна без переходів між жестами
та класу 0, тож оцінка, ймовірно, оптимістична. Деталі — в аудиті.

## 1D-CNN

```bash
python experiments/train_cnn.py              # fs=200 Гц, вікно [40, 8]
python experiments/train_cnn.py --fs 1000    # сітка як у baseline, вікно [200, 8]
```

Модель — 3 блоки Conv1d → ReLU → MaxPool, Dropout 0,3 і повнозв'язний шар
(34 086 параметрів при fs=200). Розбиття те саме, що в baseline (`results/split.json`).
Найкраща епоха обирається за val macro-F1, як і гіперпараметри baseline.
Чекпоінт `models/best_cnn.pt` містить ваги, архітектуру, нормалізацію та
відповідність класів. Для використання потрібен лише `models/cnn.py`:

```python
from models.cnn import load_cnn, predict_window
model, meta = load_cnn("models/best_cnn.pt")
label = predict_window(model, meta, window)   # window: [40, 8], відфільтроване
```

Результат на test (`results/cnn.csv`): CNN_1D — accuracy 0,771, macro-F1 0,767;
CNN_1D_fs1000 — 0,768 / 0,764. Тобто на рівні класичних моделей. Val loss росте
з 3-ї епохи, хоча val F1 тримається на рівні ~0,80, тож на нових піддослідних
мережа перенавчається.

## Експерименти

```bash
python experiments/run_experiments.py --estimate-only   # оцінка часу
python experiments/run_experiments.py --quick           # швидка перевірка (results/quick/, не в git)
python experiments/run_experiments.py                   # усі три (~5–10 хв на CPU)
```

Кожен запуск дописує рядки в `results/experiment_log.csv`. Гіперпараметри SVM
беруться з `results/baseline.csv`; test для налаштування не використовується.

1. **SVM проти CNN** на трьох розбиттях 24/6/6 (зерна 0, 1, 2), `results/exp1.csv`:
   macro-F1 SVM 0,846 ± 0,030, CNN 0,816 ± 0,032. SVM кращий в усіх трьох розбиттях.
2. **Свій проти чужого** для 6 test-піддослідних, тест — серія 2 (`results/exp2.csv`,
   `figures/exp2_own_vs_cross.png`). Навчання на власній серії 1 дає в середньому на
   13–14 пунктів macro-F1 більше, ніж навчання на 24 інших людях (SVM 0,861 проти 0,731,
   CNN 0,859 проти 0,720). Серії, ймовірно, записано в одній сесії, тож «свій» результат
   оптимістичний.
3. **Розмір вікна** 150/200/300 мс, CNN із global average pooling (`results/exp3.csv`,
   `figures/exp3_window_size.png`). Test macro-F1 0,762 / 0,750 / 0,757, тобто різниця
   в межах шуму; затримку майже повністю визначає довжина вікна.

## Тести

```bash
python -m pytest
```

93 тести: синтетичні сигнали з відомими значеннями (константа, синус, пилка),
регресійні тести на знайдені в даних проблеми та кілька тестів на реальних
файлах. Тести на реальних файлах пропускаються, якщо `data/` немає.

## Правила проєкту

- Train/val/test розбиваються **лише за піддослідними**. Нормалізацію вчимо тільки на train.
- Зерно випадковості фіксоване. Результати експериментів зберігаються в `results/` (CSV + PNG).
- Код пишемо з типами, докстрінгами (українською) і pytest-тестами.

## Відомі обмеження

- Смуга до 450 Гц на цих даних пропускає здебільшого артефакти утримання значень.
  ZC і SSC рахують переважно відгук фільтра на «сходинки», а не ЕМГ. Варіант із
  прорідженням до ~100 Гц ще не досліджено.
- Причина неузгодженості розмітки з активністю не з'ясована.
- NinaPro та Myo Armband Dataset ще не завантажені й не підтримуються.
