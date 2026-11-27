# Обучение сплатов на Python

Прототип запускается из корня репозитория и читает сохранённый приложением `.pscan`. SlangPy компилирует исходники из [каталога шейдеров Android](../../android/app/src/main/cpp/vk/shaders) при старте. Python связывает буферы по именам, заменяя в памяти объявления Vulkan-привязок и push constants. Проекция, GPU scan и radix sort, растеризация, обратные проходы, SSIM, лосс глубины, билатеральная сетка и Adam используют те же шейдеры, что и приложение.

## Запуск

Python 3.10+, GPU с Metal или Vulkan. На macOS нужен Xcode с установленным Metal Toolchain. По умолчанию выбирается Metal на macOS и Vulkan на остальных системах. Подробности установки есть в [документации SlangPy](https://slangpy.shader-slang.org/en/stable/).

```bash
python3 -m venv tools/splat_training/.venv
source tools/splat_training/.venv/bin/activate
python -m pip install -r tools/splat_training/requirements.txt
python -m pip install tensorboard  # Для --tensorboard

python -m tools.splat_training "$HOME/Downloads/scan-20261122-171427.pscan" \
  --output /tmp/pocket-splat-debug --epochs 2 --max-frames 12 \
  --validation-every 1 --tensorboard
```

`--max-frames` берёт первые N кадров по ID после фильтрации по размеру. Уберите его для полного набора. По умолчанию обучение идёт 100 эпох с бюджетом 3 млн гауссиан. Для validation выделяется 5% кадров, проход выполняется каждые 5 эпох. Для нового запуска укажите новый каталог `--output`. Если в архиве нет `mesh.ply`, передайте бинарный PLY, экспортированный из приложения, через `--mesh`.

Посев использует размер вокселя из `comment voxel` в PLY. Кадры декодируются по одному, а Z16 переводится в метры по `depthScale` из манифеста. Validation обновляет только сетку своего кадра и измеряет цвет после обновления. Train измеряет цвет до обновления гауссиан и сетки.

## Отсев кадров

Отсев включён по умолчанию. Для прогона на всех train-кадрах добавьте `--no-filter-redundant-frames`. Набор validation выбирается перед отсевом, затем расписания Adam рассчитываются по оставшимся кадрам.

Тренер проецирует вершины и центры треугольников меша в train-кадры и сверяет их с Z16. Сначала он проверяет кандидатов из групп близких поз, отдавая приоритет на удаление кадрам с худшим совпадением глубины и меньшей резкостью. Затем проверяет остальные кадры. Для каждой точки с двумя и более наблюдениями сохраняются как минимум два. Если у точки было одно наблюдение, оно сохраняется. Если точку видела пара камер с базой от 5 см и углом между лучами от 3°, сохраняется хотя бы одна такая пара. Кадры без подтверждённых точек остаются в обучении.

В `run.json` сохраняются исходное количество train-кадров, ID исключённых кадров, ID кадров без подтверждений и количество точек с измеренным покрытием и параллаксом.

## Результаты и отладка

- `metrics.jsonl` — L1, SSIM, PSNR, inverse-depth L1 и число гауссиан в конце каждого прохода.
- `run.json` и `hyperparameters.json` — ID кадров, настройки и горизонты расписаний.
- `val/<id>.png` — рендер с обновлённой сеткой текущего кадра validation.
- `scene.ply` — итоговые гауссианы в формате экспорта Android.
- `tensorboard/` — при `--tensorboard` содержит метрики по кадрам и эпохам, скорости обучения и количество пересечений с тайлами.

```bash
tensorboard --logdir /tmp/pocket-splat-debug/tensorboard
python -m unittest tools.splat_training.test_training
```

Для экспериментов передайте `--hyper settings.json` с нужными полями из [SplatHyper](../../android/app/src/main/cpp/vk/splat_train.h), например `{"densifyFrom": 5, "densifyInterval": 5, "gradThreshold": 0.0002}`. При стандартных настройках первое изменение плотности происходит на 500-м train-шаге. Для пошаговой отладки начните с `Trainer.step` в [trainer.py](trainer.py). Посев, clone/split/prune и сброс непрозрачности находятся в [population.py](population.py), управление GPU-буферами и сортировка — в [gpu.py](gpu.py).
