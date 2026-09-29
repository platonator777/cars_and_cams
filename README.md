# Vehicle ReID: поиск автомобиля между камерами

Решение формирует визуальный признак автомобиля по кадру и заданному BBox, ищет похожие изображения в статичной галерее и может отказаться от неуверенного совпадения. Детекция, OCR номеров и обработка видео в пайплайн не входят. 

Для проверки достаточно пакетного запуска: на вход подаются каталог кадров и два CSV, на выходе создаются `submission.csv`, `embeddings.npy` и `candidates.csv`. API и БД отсутствуют.

## Структура проекта

```text
final_solution/
├── README.md                         # эта инструкция
├── Dockerfile                        # образы для инференса и обучения
├── docker-compose.yml                # запуск инференса с GPU
├── .dockerignore                     # исключения из контекста сборки
├── .gitattributes                    # хранение весов через Git LFS
├── .gitignore                        # исключение презентации и временных файлов
├── pyproject.toml                    # метаданные Python-проекта
├── requirements-runtime.lock         # зависимости инференса
├── requirements-train.lock           # зависимости обучения
├── infer.py                          # пакетный инференс
├── train.py                          # обучение модели
├── evaluate.py                       # запуск локальной оценки
├── verify.py                         # проверка формата результатов
├── benchmark.py                      # замер скорости извлечения признаков
├── configs/
│   ├── inference.yaml                # параметры релизного инференса
│   ├── training/dinov2_vitb14.json   # параметры обучения
│   └── protocols/fold0/              # фиксированный локальный протокол
│       ├── fold_0_manifest.json
│       ├── query.csv
│       ├── gallery.csv
│       ├── ground_truth.csv
│       └── protocol.json
│ 
├── official/
│   └── evaluate.py                   # официальный оценщик
├── scripts/
│   ├── fetch_training_assets.py      # загрузка исходных весов DINOv2
│   ├── reproduce.py                  # воспроизведение обучения
│   └── verify_streaming_independence.py  # проверка независимости запросов
├── submission/                       # готовый результат на опубликованном тесте
│   ├── submission.csv
│   ├── embeddings.npy
│   └── candidates.csv
├── vehicle_reid/                     # реализация модели и поиска
│   ├── __init__.py
│   ├── extraction.py
│   ├── protocol.py
│   ├── scoring.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── dino.py
│   └── training/
│       ├── __init__.py
│       └── data.py
├── vendor/dinov2/                    # включенный исходный код DINOv2
│   ├── dinov2/                       # upstream-пакет и его модули
│   ├── hubconf.py
│   ├── LICENSE
│   ├── LICENSE_CELL_DINO_CODE
│   ├── LICENSE_CELL_DINO_MODELS
│   └── LICENSE_XRAY_DINO_MODEL
└── weights/
    └── model.pt                     # готовые веса для инференса
```


## Быстрый запуск на данных организаторов

Нужны Docker Compose и NVIDIA GPU, доступный контейнерам через `--gpus all`. Сборка образа может использовать сеть; **запуск инференса проходит без сети**. Выполняйте команды из каталога `final_solution`.

`INPUT_DIR` — **абсолютный путь на вашей машине** к каталогу тестовых данных, которые предоставят организаторы. Это отдельный каталог вне `final_solution`;  Например, если данные распакованы в `/srv/reid-test`, структура должна быть такой:

```text
/srv/reid-test/
├── images/
│   ├── car_0001.jpg
│   ├── car_0002.jpg
│   └── ...
├── test_query.csv
└── test_gallery.csv
```

`OUTPUT_DIR` — другой абсолютный путь на вашей машине, куда будут записаны три результата. В примере ниже это `/srv/reid-results`. Замените оба пути на реальные пути вашего сервера:

```bash
cd final_solution
docker compose build infer

export INPUT_DIR=/srv/reid-test
export OUTPUT_DIR=/srv/reid-results
docker compose run --pull never --rm infer
```

`OUTPUT_DIR` должен быть новым или пустым. Compose монтирует `INPUT_DIR` внутри контейнера как `/data` только для чтения, а `OUTPUT_DIR` как `/output`. Образ содержит готовые веса, а контейнер запускается с `network_mode: none`; загрузка модели или пакетов во время инференса не требуется.

## Результаты и проверка формата

| Файл | Содержимое |
|---|---|
| `submission.csv` | Без заголовка: `query_id,gallery_id_1,...,gallery_id_10`. По одной строке и десять разных кандидатов на каждый запрос в порядке `test_query.csv`. |
| `embeddings.npy` | Матрица `float32` размера `(число query + число gallery, 384)`: сначала query, затем gallery, в порядке входных CSV. |
| `candidates.csv` | Заголовок `query_id,gallery_id,confidence`. Не более одного принятого top-1 на запрос; отсутствие строки означает отказ. |

Проверка трех файлов в уже собранном образе:

```bash
docker compose run --pull never --rm --entrypoint python infer \
  verify.py --output /output \
  --query /data/test_query.csv --gallery /data/test_gallery.csv
```

Проверка метрик при наличии `ground_truth.csv` в `INPUT_DIR`:

```bash
docker compose run --pull never --rm --entrypoint python infer \
  evaluate.py --gt /data/ground_truth.csv \
  --submission /output/submission.csv \
  --candidates /output/candidates.csv \
  --embeddings /output/embeddings.npy \
  --query /data/test_query.csv --gallery /data/test_gallery.csv \
  --json /output/metrics.json
```

## Модель и выбранный порог

`DINOv2 ViT-B/14` получает crop автомобиля с 5% контекста, квадратным дополнением и размером 336×336. CLS-признак проецируется в 384-мерный L2-нормированный вектор. Для каждого запроса выбираются top-50 объектов галереи по cosine similarity, затем только этот список уточняется локальным k-reciprocal rerank (`k1=15`, `k2=3`, `lambda=0.64`). Другие query при поиске не используются.

В `submission.csv` сохраняется top-10 независимо от решения об отказе. В `candidates.csv` попадает top-1 только при rerank-score **≥ 0.7469556331634521**. Порог выбран на отдельной локальной calibration-части по `0.7 × F1 + 0.3 × TNR`. Значение, параметры модели и SHA-256 весов зафиксированы в [`configs/inference.yaml`](configs/inference.yaml).

Размер `weights/model.pt` — 349 470 663 байта; SHA-256: `092f96bed90d40ecdce48a4ff53e6ea3e52a4b55e681a0bac07092dfb3042310`.

Локальная проверка на fold 0 дала mAP@10 `0.649523`, Rank-1 `0.574355`, Rank-5 `0.817906` при оценке сохраненных признаков обучения. Отдельный полный прогон от изображений дал mAP@10 `0.649209`. На отложенной части для проверки фиксированного порога получены F1 `0.88911` и TNR `0.97241`.

## Воспроизведение обучения

Обучение требует CUDA, исходный набор соревнования с `train.csv` и `images/`, а также публичную инициализацию DINOv2-B/14. Публичные веса **не нужны для инференса**: дообученный checkpoint уже включен в `weights/`. Для обучения загрузите и проверьте исходные веса один раз:

```bash
python3 scripts/fetch_training_assets.py \
  --assets /absolute/path/to/training_assets --model b14
```

Скрипт скачивает файл `dinov2_vitb14_pretrain.pth` и проверяет SHA-256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`. Если файл с этим хешем уже лежит в каталоге assets, сеть не нужна.

Запуск обучения в контейнере:

```bash
docker build --target training -t vehicle-reid-train:locked .
mkdir -p /absolute/path/to/runs

docker run --rm --gpus all --shm-size 2g --network none \
  -v /absolute/path/to/train_data:/data:ro \
  -v /absolute/path/to/training_assets:/assets:ro \
  -v /absolute/path/to/runs:/runs \
  vehicle-reid-train:locked \
  --config /app/configs/training/dinov2_vitb14.json \
  --data-root /data --assets /assets --output /runs/repeat-01
```

`/runs/repeat-01` не должен существовать до запуска. Обучение использует зафиксированный fold 0: 7 620 изображений и 1 232 ID для обновления весов; 1 936 изображений и 309 непересекающихся ID для выбора эпохи по официальному mAP@10. Первые две эпохи обучается голова, затем последние два блока backbone и голова; используются AdamW, cross entropy с label smoothing и batch-hard triplet loss.

Итоги нового прогона: `/runs/repeat-01/result.json`, `/runs/repeat-01/artifacts/best_model.pt`, история эпох и точные файлы официальной оценки. Прогон не заменяет релизные `weights/model.pt`. Для возобновления **незавершенного** прогона повторите ту же команду с `--output /runs/repeat-01 --resume /runs/repeat-01/resume.pt`.

## Замер производительности

После запуска инференса можно измерить только извлечение признака на тех же кадрах:

```bash
docker compose run --pull never --rm --entrypoint python infer \
  benchmark.py --config /app/configs/inference.yaml \
  --images /data/images --query /data/test_query.csv \
  --report /output/benchmark.json
```

Скрипт включает чтение, декодирование, crop, preprocessing, forward и L2-нормализацию; поиск и rerank не входят в latency. По умолчанию выполняются 50 прогревов, 300 измерений batch=1 и замеры FPS при batch 1/8/16/32. Нужно не менее 32 query. Локально на RTX 3060 Ti получены p50 54.72 мс и лучший FPS 97.86.

## Состав пакета и внешние источники

| Путь | Назначение |
|---|---|
| `infer.py`, `vehicle_reid/`, `configs/inference.yaml`, `weights/model.pt` | Рабочий инференс. |
| `Dockerfile`, `docker-compose.yml`, `requirements-runtime.lock`, `pyproject.toml` | Сборка и зависимости. |
| `verify.py`, `evaluate.py`, `official/evaluate.py`, `benchmark.py` | Проверка формата, официальные метрики и скорость. |
| `train.py`, `scripts/reproduce.py`, `scripts/fetch_training_assets.py`, `configs/training/`, `configs/protocols/fold0/` | Воспроизведение обучения. |
| `submission/` | Готовые файлы для опубликованного теста; для закрытого теста запускайте `infer.py` заново. |
