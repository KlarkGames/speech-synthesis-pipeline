# Emotional Speech Synthesis Pipeline

# Описание проекта

Данный репозиторий является пайплайном обучения Emotional Speech Synthesis модели для одноголосой озвучки диалогов, генерируемых LLM.

Подробнее про задачу, для которой разрабатывается данный пайплайн можно прочесть в [ML System Design документе](docs/ml_system_design_doc.md).

# Содержание

- [Emotional Speech Synthesis Pipeline](#emotional-speech-synthesis-pipeline)
- [Описание проекта](#описание-проекта)
- [Содержание](#содержание)
- [Формирование стандартизированного формата](#формирование-стандартизированного-формата)
  - [Предобработка датасета MLS](#предобработка-датасета-mls)
  - [Предобработка датасета EmoV\_DB](#предобработка-датасета-emov_db)
  - [Предобработка директории с аудиофайлами](#предобработка-директории-с-аудиофайлами)
- [Работа с объектным хранилищем S3 и LakeFS](#работа-с-объектным-хранилищем-s3-и-lakefs)
  - [Конфигурация](#конфигурация)
  - [Загрузка данных](#загрузка-данных)
    - [Загрузка любой директории](#загрузка-любой-директории)
- [Сохранение метаданных о датасетах](#сохранение-метаданных-о-датасетах)
  - [Требуемая структура реляционной базы данных](#требуемая-структура-реляционной-базы-данных)
  - [Сбор аудио метаданных](#сбор-аудио-метаданных)
  - [Сбор текстовых метаданных](#сбор-текстовых-метаданных)
  - [Распознование произнесенного текста с помощью ASR](#распознование-произнесенного-текста-с-помощью-asr)
    - [Обработка датасета с помощью ASR](#обработка-датасета-с-помощью-asr)
  - [Вычисление WER/CER между ASR и Original текстами](#вычисление-wercer-между-asr-и-original-текстами)
  - [Улучшение качества с помощью Resemble Enhancer'а](#улучшение-качества-с-помощью-resemble-enhancerа)
    - [Запуск Enhancer'а](#запуск-enhancerа)
    - [Обработка Enhancer'ом стандартизированного датасета:](#обработка-enhancerом-стандартизированного-датасета)
  - [Получение информации о интонационных паузах с помощью Montreal Forced Aligner](#получение-информации-о-интонационных-паузах-с-помощью-montreal-forced-aligner)
    - [Установка зависимостей](#установка-зависимостей)
    - [Обработка датасета с помощью MFA](#обработка-датасета-с-помощью-mfa)
  - [Фильтрация датасетов по собранным метаданным. Формирование filtered\_metadata.csv](#фильтрация-датасетов-по-собранным-метаданным-формирование-filtered_metadatacsv)

# Формирование [стандартизированного формата](docs/ml_system_design_doc.md#32-стандартизованный-формат-хранения-данных)

## Предобработка датасета MLS

Чтобы предобработать датасет скачайте и распакуйте его в директорию (Например *data/raw/mls_oput_english*) и запустите скрипт:

```
python -m src.data.MLS.preprocess --dataset-path [DATASET_DIRECTORY] --output-path [SAVE_DIRECTORY] --change-sample-rate True --result-sample-rate 44100 --n-files 3600
```

Результатом будет сформатированный под [принятую структуру](docs/ml_system_design_doc.md#32-стандартизованный-формат-хранения-данных) датасет в директории сохранения.

Описание всех параметров представлено ниже:
- **--dataset-path** - Path to MLS dataset
- **--output-path** - Path to output directory
- **--change-sample-rate** - Resample all audiofiles to specified sample rate. *Default: False*
- **--result-sample-rate** - Resample all audiofiles to specified sample rate. *Default: 44100*
- **--n-jobs** - Number of parallel jobs. If set to -1, use all available CPU cores. *Default: -1*
- **--cache-dir** - Directory in output path to store cache files. *Default: .cache*
- **--n-files** - Number of files to process. If set to -1, process all files of speaker. Mean duration of files is 15s, so if you want to process 1h of speech, set this to 3600. If there not enough files, all files will be processed. *Default: 3600*
- **--cache-every-n-speakers** - Number of speakers to be processed before cache is updated. *Default: 100*

## Предобработка датасета EmoV_DB

Чтобы предобработать датасет скачайте и распакуйте его в директорию (Например *data/raw/EmoV_DB*) и запустите скрипт:

```
python -m src.data.EmoV_DB.preprocess --dataset-path [DATASET_DIRECTORY] --output-path [SAVE_DIRECTORY] --change-sample-rate True --result-sample-rate 44100 --download-cmuarctic-data True
```

Результатом будет сформатированный под [принятую структуру](#структура-датасетов-после-обработки) датасет в директории сохранения.

Описание всех параметров представлено ниже:
- **--dataset-path** - Path to EmoV_DB dataset
- **--output-path** - Path to output directory
- **--cmuarctic-data-path** - Path to 'cmuarctic.data' file with texts for audiofiles. *Default: None*
- **--cmuarctic-url** - Url to 'cmuarctic.data' file to be able to download this file if it doesn't exist. *Default: http://www.festvox.org/cmu_arctic/cmuarctic.data*
- **--download-cmuarctic-data** - Download 'cmuarctic.data' file if it doesn't exist to input dataset path. *Default: False*
- **--change-sample-rate** - Resample all audiofiles to specified sample rate. *Default: False*
- **--result-sample-rate** - Resample all audiofiles to specified sample rate. *Default: 44100*
- **--n-jobs** - Number of parallel jobs. If set to -1, use all available CPU cores. *Default: -1*

## Предобработка директории с аудиофайлами

Если имеются неструктурированные аудиофайлы, находящиеся в одной директории, их можно сформировать в датасет следующим скриптом:

```
python -m src.datasets.audio_folder --folder-path [PATH_TO_DIRECTORY_WITH_AUDIOFILES] --save-path [STRUCTURED_DATASET_SAVE_PATH]
```

Допустим есть следующий датасет:

```
raw_audio_path/
├── audio_1.mp3
├── some_directory/
│   ├── audio_2.ogg
│   └── another_directory/
│       └── audio_3.flac
└── and_another_one_directory/
    └── audio_4.wav
```

Обычный запуск скрипта создаст следующую структуру:

```
dataset_path/
├── wavs/
│   └── audio_1.wav
├── speaker_0/
│   └── wavs/
│       ├── audio_2.wav
│       └── another_directory/
│           └── audio_3.wav
├── speaker_1/
│   └── wavs/
│       └── audio_4.wav
└── metadata.csv
```

`metadata.csv` будет содержать следующую информацию:

| path_to_wav                                  | speaker_id |
| -------------------------------------------- | ---------- |
| wavs/audio_1.wav                             | -1         |
| speaker_0/wavs/audio_2.wav                   | 0          |
| speaker_0/wavs/another_directory/audio_3.wav | 0          |
| speaker_1/wavs/audio_4.wav                   | 1          |

Если указать флаг `--unknown-speaker` добавит все аудиофайлы со speaker_id -1:

```
dataset_path/
├── wavs/
│   ├── audio_1.wav
│   ├── some_directory/
│   │   ├── audio_2.wav
│   │   └── another_directory/
│   │       └── audio_3.wav
│   └── and_another_one_directory/
│       └── audio_4.wav
└── metadata.csv
```

`metadata.csv` будет содержать следующую информацию:

| path_to_wav                                       | speaker_id |
| ------------------------------------------------- | ---------- |
| wavs/audio_1.wav                                  | -1         |
| wavs/some_directory/audio_2.wav                   | -1         |
| wavs/some_directory/another_directory/audio_3.wav | -1         |
| wavs/and_another_one_directory/audio_4.wav        | -1         |

Если указать флаг `--single-speaker` все аудиофайлы будут считаться от одного спикера:

```
dataset_path/
├── speaker_0/
│   └── wavs/
│       ├── audio_1.wav
│       ├── some_directory/
│       │   ├── audio_2.wav
│       │   └── another_directory/
│       │       └── audio_3.wav
│       └── and_another_one_directory/
│           └── audio_4.wav
└── metadata.csv
```

`metadata.csv` будет содержать следующую информацию:

| path_to_wav                                                 | speaker_id |
| ----------------------------------------------------------- | ---------- |
| speaker_0/wavs/audio_1.wav                                  | 0          |
| speaker_0/wavs/some_directory/audio_2.wav                   | 0          |
| speaker_0/wavs/some_directory/another_directory/audio_3.wav | 0          |
| speaker_0/wavs/and_another_one_directory/audio_4.wav        | 0          |

**It's imposible** to specify both `--single-speaker` and `--unknown-speaker` flags. You'll get ValueError.

Все параметры указаны ниже:
- **--folder-path** - Path to the folder with audio files.
- **--single-speaker** - Is all files in this folder belongs to one speaker? Default: False
- **--unknown-speaker** - Is all files in this folder belongs to unknown speaker? Default: False
- **--overwrite** - Is it needed to overwrite existing files? Default: False
- **--save-path** - Path where to save formated dataset.
- **--n-jobs** - Number of parallel jobs to use while processing. -1 means to use all cores. Default: -1

# Работа с объектным хранилищем S3 и LakeFS

## Конфигурация

Данный проект позволяет использовать объектные хранилища для сохранения датасетов. Для возиожности версионирования данных предлагается использовать поверх объектного хранилища LakeFS. Многие скрипты поддерживают работу с LakeFS с помощью бибилиотеки [`lakefs-spec`](https://github.com/aai-institute/lakefs-spec).

Все скрипты, работающие с LakeFS прмнмиают следующй набор параметров:

```
LAKEFS_ADDRESS=http://[YOUR_IP_ADDRESS]
LAKEFS_PORT=[YOUR_LAKEFS_PORT]
LAKEFS_ACCESS_KEY_ID=[YOUR_ACCESS_KEY_ID]
LAKEFS_SECRET_KEY=[YOUR_SACRET_KEY]
```

Все эти параметры можно указать в соответствующих CLI аргументах, но чтобы этого не делать постоянно - рекомендуется прописать эти параметры в `.env` файле.

## Загрузка данных

### Загрузка любой директории

Для возможности загрузки любых файлов можно воспользоваться или [lakefsclt](https://docs.lakefs.io/reference/cli.html), или скриптом `src/datasets/load_diretory_to_lakefs.py`

```
python -m src.datasets.load_directory_to_lakefs --path [PATH_TO_YOUR_DIRECTORY] --repository-name [NAME_OF_YOUR_REPOSITORY]
```

# Сохранение метаданных о датасетах

## Требуемая структура реляционной базы данных

Для полноценного функционирования скриптов обработки требуется поднять PostgreSQL со следующей схемой:

![alt text](docs/images/Metadata_DB.drawio.png)

Всю информацию о базе данных данных следует сохранить в `.env` файл, из которого скрипты автоматически подтянут их без надобности указывать в аргументах скриптов.

```
POSTGRES_USER=[POSTGRES_USER]
POSTGRES_PASSWORD=[POSTGRES_PASSWORD]
POSTGRES_DB=[POSTGRES_DB_NAME]
POSTGRES_PORT=5432
POSTGRES_ADDRESS=localhost
```

## Сбор аудио метаданных

Для того, чтобы собрать метаданные с стандартизированного датасета, воспользуйтесь следующей командой:

```
python -m src.metrics_collection.collect_audio_metrics local --dataset-path [PATH_TO_DATASET]
```

Скрипт пробежит по всем файлам, которые не были добавлены в базу данных ранее, и сохранит их аудио метаданные. Так же скрипт запишет информацию о пренадлежности файлов к добавляемому датасету.

Скрипт высчитывает хеш файлов в качестве идентификатора, что гарантирует, что даже если файл был добавлен в несколько датасетов, при обработке каждого из них файл будет обработан единажды (Если конечно не прописать `--overwrite`, что заставит скрипт перезаписать информацию в БД).

Все параметры указаны ниже:

- **Общие**:
  - **--metadata_path** - Path to .csv file with metadata. Default: [DATASET_PATH]/metadata.csv
  - **--overwrite** - Is to overwrite existing metrics or not. Default: False
  - **--database-address** - Address of the database. Environment Variable: POSTGRES_ADDRESS
  - **--database-port** - Port of the database. Environment Variable: POSTGRES_PORT
  - **--database-user** - Username to use for database authentication. Environment Variable: POSTGRES_USER
  - **--database-password** - Password to use for database authentication. Environment Variable: POSTGRES_PASSWORD
  - **--database-name** - Name of the database. Environment Variable: POSTGRES_DB
  - **--n-jobs** - Number of parallel jobs to use while processing. -1 means to use all cores. Default: -1

- **local**:
  - **--dataset-path** - Path to data to process.

- **s3**:
  - **--LakeFS-address** - LakeFS address. Environment Variable: LAKEFS_ADDRESS
  - **--LakeFS-port** - LakeFS port. Environment Variable: LAKEFS_PORT
  - **--ACCESS-KEY-ID** - Access key id of LakeFS. Environment Variable: LAKEFS_ACCESS_KEY_ID
  - **--SECRET-KEY** - Secret key of LakeFS. Environment Variable: LAKEFS_SECRET_KEY
  - **--repository-name** - Name of LakeFS repository
  - **--branch-name** - Name of the branch. Default: main

## Сбор текстовых метаданных

Данный скрипт добавляет соответствия между аудио файлом и произносимому тексту. Для этого скрипта **требуется наличие поля "text"** в `metadata.csv`. Если такого поля не будет найдено - скрипт выбросит ошибку.

Для заргузки текста в базу данных выполните следующую команду:
```
python -m src.metrics_collection.collect_audio_text local --dataset-path [PATH_TO_DATASET]
```

Все параметры указаны ниже:
- **Общие**:
  - **--metadata_path** - Path to .csv file with metadata. Default: [DATASET_PATH]/metadata.csv
  - **--overwrite** - Is to overwrite existing metrics or not. Default: False
  - **--database-address** - Address of the database. Environment Variable: POSTGRES_ADDRESS
  - **--database-port** - Port of the database/ Environment Variable: POSTGRES_PORT
  - **--database-user** - Username to use for database authentication. Environment Variable: POSTGRES_USER
  - **--database-password** - Password to use for database authentication. Environment Variable: POSTGRES_PASSWORD
  - **--database-name** - Name of the database. Environment Variable: POSTGRES_DB
  - **--n-jobs** - Number of parallel jobs to use while processing. -1 means to use all cores. Default: -1

- **local**:
  - **--dataset-path** - Path to data to process.

- **s3**:
  - **--LakeFS-address** - LakeFS address. Environment Variable: LAKEFS_ADDRESS
  - **--LakeFS-port** - LakeFS port. Environment Variable: LAKEFS_PORT
  - **--ACCESS-KEY-ID** - Access key id of LakeFS. Environment Variable: LAKEFS_ACCESS_KEY_ID
  - **--SECRET-KEY** - Secret key of LakeFS. Environment Variable: LAKEFS_SECRET_KEY
  - **--repository-name** - Name of LakeFS repository
  - **--branch-name** - Name of the branch. Default: main

## Распознование произнесенного текста с помощью ASR

Для оценки качества голоса в записях используется оценка с помощью Automatic Speech Recognition. Оригинальные текст сопоставляется с распознанным по WER и CER. В случаях, когда эти показатели превышают требуемый порог - информация о семплах удаляется из `metadata.csv`.

Для анализа используется Triton Inference Server с ASR моделью. Данный сервер находится под NDA. Принцип работы аналогичен [обработке Enhancer'ом](#улучшение-качества-с-помощью-resemble-enhancerа).

### Обработка датасета с помощью ASR

После поднятия ASR Triton Inference Server'a запустите следующий скрипт:

```
python -m src.preprocessing.asr_processing --triton-port 127.0.0.1 --triton-port 9870 local --dataset-path [PATH_TO_DATASET] 
```

Описание всех параметров представлено ниже:
- **Общие**:
  - **--metadata-path** - Path to .csv file with metadata. Default: [DATASET_PATH]/metadata.csv
  - **--triton-address** - Address of the Triton Inference Server. Default: localhost
  - **--triton-port** - Port of the Triton Inference Server. Default: 8000
  - **--batch-size** - Batch size for processing audio files. Default: 10
  - **--overwrite** - Is to overwrite existing metrics or not. Default: False
  - **--database-address** - Address of the database. Environment Variable: POSTGRES_ADDRESS
  - **--database-port** - Port of the database/ Environment Variable: POSTGRES_PORT
  - **--database-user** - Username to use for database authentication. Environment Variable: POSTGRES_USER
  - **--database-password** - Password to use for database authentication. Environment Variable: POSTGRES_PASSWORD
  - **--database-name** - Name of the database. Environment Variable: POSTGRES_DB
  - **--n-jobs** - Number of parallel jobs to use while processing. -1 means to use all cores. Default: -1

- **local**:
  - **--dataset-path** - Path to data to process.

- **s3**:
  - **--LakeFS-address** - LakeFS address. Environment Variable: LAKEFS_ADDRESS
  - **--LakeFS-port** - LakeFS port. Environment Variable: LAKEFS_PORT
  - **--ACCESS-KEY-ID** - Access key id of LakeFS. Environment Variable: LAKEFS_ACCESS_KEY_ID
  - **--SECRET-KEY** - Secret key of LakeFS. Environment Variable: LAKEFS_SECRET_KEY
  - **--repository-name** - Name of LakeFS repository
  - **--branch-name** - Name of the branch. Default: main

## Вычисление WER/CER между ASR и Original текстами

Для обнарушения несоответствий между произносимым и размеченными текстами полезно вычислить WER/CER для тех файлов, которые имеют оба текста.

Чтобы добавить эту информацию в базу данных, выполните:
```
python -m src.metrics_collection.calculate_wer_cer local --dataset-path [PATH_TO_DATASET]
```

Семплы, для которых нет одного из текстов будут проигнорированы и WER/CER для них не будет подсчитан. Отмечу, что при изменении одного из текстов в базе данных, WER/CER автоматически не будет пересчитан, поэтому полезным может быть иногда перезаписывать данные с помощью `--overwrite`.

Все параметры указаны ниже:
- **Общие**:
  - **--metadata_path** - Path to .csv file with metadata. Default: [DATASET_PATH]/metadata.csv
  - **--overwrite** - Is to overwrite existing metrics or not. Default: False
  - **--database-address** - Address of the database. Environment Variable: POSTGRES_ADDRESS
  - **--database-port** - Port of the database/ Environment Variable: POSTGRES_PORT
  - **--database-user** - Username to use for database authentication. Environment Variable: POSTGRES_USER
  - **--database-password** - Password to use for database authentication. Environment Variable: POSTGRES_PASSWORD
  - **--database-name** - Name of the database. Environment Variable: POSTGRES_DB
  - **--n-jobs** - Number of parallel jobs to use while processing. -1 means to use all cores. Default: -1

- **local**:
  - **--dataset-path** - Path to data to process.

- **s3**:
  - **--LakeFS-address** - LakeFS address. Environment Variable: LAKEFS_ADDRESS
  - **--LakeFS-port** - LakeFS port. Environment Variable: LAKEFS_PORT
  - **--ACCESS-KEY-ID** - Access key id of LakeFS. Environment Variable: LAKEFS_ACCESS_KEY_ID
  - **--SECRET-KEY** - Secret key of LakeFS. Environment Variable: LAKEFS_SECRET_KEY
  - **--repository-name** - Name of LakeFS repository
  - **--branch-name** - Name of the branch. Default: main

## Улучшение качества с помощью Resemble Enhancer'а

Для улучшения качества голоса и избавления от внешних шумов используется [resemble-enhancer](https://github.com/resemble-ai/resemble-enhance/tree/main). Данный инструмент инкапсулирован в Triton Inference Server, подробнее про сервер можно прочитать в [официальных туториалах](https://github.com/triton-inference-server/tutorials). 

Данный инструмент обрабатывает **только .wav файлы** и результатом являются .wav файлы с частотой дикретизации **44.1kHZ** с одним каналом (**mono**).


Для обработки требуется сперва поднять Docker контейнер с Enhancer'ом

### Запуск Enhancer'а

```
docker compose -f triton/enhancer/compose.yaml up
```

Можно изменить порты Triton'а или количество доступных видеокарт прописав соответствующие переменные среды в `.env` файле. Пример приведет в `.env.example`

```
docker compose --env-file .env -f triton/enhancer/compose.yaml up
```

Порты по умолчанию:
- TRITON_HTTP_PORT: 8520
- TRITON_GRPC_PORT: 8521
- TRITON_METRICS_PORT: 8522

### Обработка Enhancer'ом [стандартизированного датасета](#структура-датасетов-после-обработки):

```
python -m src.preprocessing.enhance --triton-address 127.0.0.1 --triton-port 8520 local_to-local --input-path [PATH_TO_ORIGIN_DATASET] --output-path [SAVE_PATH]
```

Описание всех параметров представлено ниже:
- **Общие**:
  - **--metadata-path** - Path to .csv file with metadata. Default: [DATASET_PATH]/metadata.csv
  - **--output-path** - Path where the enhanced dataset will be saved.
  - **--chunk-duration** - The duration in seconds by which the enhancer will divide your sample. Default: 30.0
  - **--chunk-overlap** - The duration of overlap between adjacent samples. Does not enlarge chunk_duration. Default: 1.0
  - **--model-name** - The name of Triton Inference Server model. Default: enhancer_ensemble
  - **--batch-size** - The size of the batch of async tasks every job will process
  - **--triton-address** - The Triton Inference Server address
  - **--triton-port** - The Triton Inference Server port
  - **--n-jobs** - Number of parallel jobs. If -1 specified, use all available CPU cores.

- **Для конфигурации s3**:
  - **--LakeFS-address** - LakeFS address. Environment Variable: LAKEFS_ADDRESS
  - **--LakeFS-port** - LakeFS port. Environment Variable: LAKEFS_PORT
  - **--ACCESS-KEY-ID** - Access key id of LakeFS. Environment Variable: LAKEFS_ACCESS_KEY_ID
  - **--SECRET-KEY** - Secret key of LakeFS. Environment Variable: LAKEFS_SECRET_KEY
  - **--LakeFS-address** - LakeFS address. Environment Variable: LAKEFS_ADDRESS
  - **--LakeFS-port** - LakeFS port. Environment Variable: LAKEFS_PORT
  - **--ACCESS-KEY-ID** - Access key id of LakeFS. Environment Variable: LAKEFS_ACCESS_KEY_ID
  - **--SECRET-KEY** - Secret key of LakeFS. Environment Variable: LAKEFS_SECRET_KEY

- **local_to_local**:
  - **--input-path** - Path to processing dataset.
  - **--output-path** - Path where the enhanced dataset will be saved.
- **local_to_s3**:
  - **--input-path** - Path to processing dataset.
  - **--output-repository-name** - Name of LakeFS repository where to store Enhanced data.
  - **--output-branch-name** - Name of the branch where to store Enhanced data. Default: main
- **s3_to_local**:
  - **--input-repository-name** - Name of LakeFS repository where processing dataset is stored.
  - **--input-branch-name** - Name of the branch where processing dataset is stored. Default: main
  - **--output-path** - Path where the enhanced dataset will be saved.
- **s3_to_s3**:
  - **--input-repository-name** - Name of LakeFS repository where processing dataset is stored.
  - **--input-branch-name** - Name of the branch where processing dataset is stored. Default: main
  - **--output-repository-name** - Name of LakeFS repository where to store Enhanced data.
  - **--output-branch-name** - Name of the branch where to store Enhanced data. 

## Получение информации о интонационных паузах с помощью Montreal Forced Aligner

Montreal Forced Aligner позволяет по имеющимся текстам и аудиофайлам получить разметку времени произнесения каждого слова/фонемы. Для последующего использования мы сохраняем эту информацию в базу данных, например для восстанавления пунктуации.

### Установка зависимостей

К сожалению Montreal Forced Aligner (MFA) имеет специфические зависимости, которые не устанавливаются корректно с помощью pip. Однако есть готовый контейнер с треубетыми библиотеками.

Чтобы поднять соответствующий контейнер используйте:

```bash
docker run -it --name MFA_Processing --network host -v [PATH_TO_DATA_TO_PROCESS]:/workspace/data -v $(pwd):/workspace mmcauliffe/montreal-forced-aligner
```

Дальшее нужно дополнительно установить библиотеки, требуемы для работы.

```
pip install textgrid, click, SQLAlchemy, psycopg, psycopg2, python-dotenv, pyyaml
```

### Обработка датасета с помощью MFA

Внутри поднятого контейнера запустите следующий скрипт

```bash
cd /workspace

python -m src.preprocessing.mfa_processing --dataset-path ./data/[YOUR_DATASET] 
```

Описание всех параметров представлено ниже:
- **--dataset-path** - Path to data to process.
- **--metadata_path** - Path to .csv file with metadata. Default: [DATASET_PATH]/metadata.csv
- **--overwrite** - Is to overwrite existing metrics or not. Default: False
- **--database-address** - Address of the database. Environment Variable: POSTGRES_ADDRESS
- **--database-port** - Port of the database/ Environment Variable: POSTGRES_PORT
- **--database-user** - Username to use for database authentication. Environment Variable: POSTGRES_USER
- **--database-password** - Password to use for database authentication. Environment Variable: POSTGRES_PASSWORD
- **--database-name** - Name of the database. Environment Variable: POSTGRES_DB
- **--n-jobs** - Number of parallel jobs to use while processing. -1 means to use all cores. Default: -1

## Фильтрация датасетов по собранным метаданным. Формирование filtered_metadata.csv

Собранные в базе данных метаданные позволяют отправлять на обработку и обучение только те данные, которые подходят нам по критериям.

Для того, чтобы сформировать для датасета новую выборку согластно конфигурационному файлу, выполните следующее:
```
python -m src.filtration.database_filtration --dataset-path [PATH_TO_DATASET] --path-to-config [PATH_TO_YAML_FILTRATION_CONFIG]
```

Передаваемый конфиг должен выглядеть как YAML файл со подобный содержанием:

```yaml
default:
  sample_rate: 44100
  channels: 1
  duration:
    min: 1
    max: 15
  SNR: 
    min: null
    max: null
  dBFS:
    min: null
    max: null
  CER: 
    min: null
    max: null
  WER: 
    min: null
    max: null
  samples_per_speaker:
    min: null
    max: null
  minutes_per_speaker:
    min: null
    max: null
  text_len_per_duration:
    min: null
    max: null
  use_unknown_speakers: True
  only_with_ASR_texts: False
  only_with_Original_texts: False
```

Самый высокий уровень содержит разные конфиги. В даннм примере конфиг один - default, и он ожидается скриптом по умолчанию. Можно указать другой конфиг с помощью параметра `--config-name`. По результатам создастся `filtered_metadata.csv` файл содержащий колонки "path_to_wav", "speaker_id", "hash".

Все параметры скрипта приведены ниже:
- **--dataset-path** - Path to data to process.
- **--path-to-config** - Path to YAML filtration config.
- **--config-name** - Name of config in YAML file to use. Default: default
- **--database-address** - Address of the database. Environment Variable: POSTGRES_ADDRESS
- **--database-port** - Port of the database/ Environment Variable: POSTGRES_PORT
- **--database-user** - Username to use for database authentication. Environment Variable: POSTGRES_USER
- **--database-password** - Password to use for database authentication. Environment Variable: POSTGRES_PASSWORD
- **--database-name** - Name of the database. Environment Variable: POSTGRES_DB
- **--save-path** - Path where to save metadata Data Frame file. Default: [DATASET_PATH]/filtered_metadata.csv
- **--include-text** - Is it needed to create "text" column in metadata file.