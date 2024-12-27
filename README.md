# Emotional Speech Synthesis Pipeline

# Описание проекта

Данный репозиторий является пайплайном обучения Emotional Speech Synthesis модели для одноголосой озвучки диалогов, генерируемых LLM.

Подробнее про задачу, для которой разрабатывается данный пайплайн можно прочесть в [ML System Design документе](docs/ml_system_design_doc.md).

# Содержание

- [Emotional Speech Synthesis Pipeline](#emotional-speech-synthesis-pipeline)
- [Описание проекта](#описание-проекта)
- [Содержание](#содержание)
- [Данные](#данные)
  - [MLS](#mls)
    - [Описание](#описание)
    - [Предобработка датасета](#предобработка-датасета)
  - [EmoV\_DB](#emov_db)
    - [Описание](#описание-1)
    - [Предобработка датасета](#предобработка-датасета-1)
- [Структура датасетов после обработки](#структура-датасетов-после-обработки)
- [Предобработка](#предобработка)
  - [Улучшение качества с помощью Resemble Enhancer'а](#улучшение-качества-с-помощью-resemble-enhancerа)
    - [Запуск Enhancer'а](#запуск-enhancerа)
    - [Обработка Enhancer'ом стандартизированного датасета:](#обработка-enhancerом-стандартизированного-датасета)
  - [Расстановка запятых и точек в местах пауз голоса с помощью Montreal Forced Aligner](#расстановка-запятых-и-точек-в-местах-пауз-голоса-с-помощью-montreal-forced-aligner)
    - [Установка зависимостей](#установка-зависимостей)
    - [Обработка датасета с помощью MFA](#обработка-датасета-с-помощью-mfa)
  - [Распознование произнесенного текста с помощью ASR](#распознование-произнесенного-текста-с-помощью-asr)

# Обработка данных

## Предобработка датасета MLS

Чтобы предобработать датасет скачайте и распакуйте его в директорию (Например *data/raw/mls_oput_english*) и запустите скрипт:

```
python -m src.data.MLS.preprocess --dataset_path [DATASET_DIRECTORY] --output_path [SAVE_DIRECTORY] --change_sample_rate True --result_sample_rate 44100 --n_files 3600
```

Результатом будет сформатированный под [принятую структуру](#структура-датасетов-после-обработки) датасет в директории сохранения.

Описание всех параметров представлено ниже:
- **--dataset_path** - Path to MLS dataset
- **--output_path** - Path to output directory
- **--change_sample_rate** - Resample all audiofiles to specified sample rate. *Default: False*
- **--result_sample_rate** - Resample all audiofiles to specified sample rate. *Default: 44100*
- **--n_jobs** - Number of parallel jobs. If set to -1, use all available CPU cores. *Default: -1*
- **--cache_dir** - Directory in output path to store cache files. *Default: .cache*
- **--n_files** - Number of files to process. If set to -1, process all files of speaker. Mean duration of files is 15s, so if you want to process 1h of speech, set this to 3600. If there not enough files, all files will be processed. *Default: 3600*
- **--cache_every_n_speakers** - Number of speakers to be processed before cache is updated. *Default: 100*

## Предобработка датасета EmoV_DB

Чтобы предобработать датасет скачайте и распакуйте его в директорию (Например *data/raw/EmoV_DB*) и запустите скрипт:

```
python -m src.data.EmoV_DB.preprocess --dataset_path [DATASET_DIRECTORY] --output_path [SAVE_DIRECTORY] --change_sample_rate True --result_sample_rate 44100 --download_cmuarctic_data True
```

Результатом будет сформатированный под [принятую структуру](#структура-датасетов-после-обработки) датасет в директории сохранения.

Описание всех параметров представлено ниже:
- **--dataset_path** - Path to EmoV_DB dataset
- **--output_path** - Path to output directory
- **--cmuarctic_data_path** - Path to 'cmuarctic.data' file with texts for audiofiles. *Default: None*
- **--cmuarctic_url** - Url to 'cmuarctic.data' file to be able to download this file if it doesn't exist. *Default: http://www.festvox.org/cmu_arctic/cmuarctic.data*
- **--download_cmuarctic_data** - Download 'cmuarctic.data' file if it doesn't exist to input dataset path. *Default: False*
- **--change_sample_rate** - Resample all audiofiles to specified sample rate. *Default: False*
- **--result_sample_rate** - Resample all audiofiles to specified sample rate. *Default: 44100*
- **--n_jobs** - Number of parallel jobs. If set to -1, use all available CPU cores. *Default: -1*

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
python -m src.preprocessing.enhance --triton-address 127.0.0.1 --triton-port 8520 --dataset-path [PATH_TO_ORIGIN_DATASET] --output-path [SAVE_PATH]
```

Описание всех параметров представлено ниже:
- **--dataset_path** - Path to processing dataset.
- **--output_path** - Path where the enhanced dataset will be saved.
- **--chunk_duration** - The duration in seconds by which the enhancer will divide your sample. Default: 30.0
- **--chunk_overlap** - The duration of overlap between adjacent samples. Does not enlarge chunk_duration. Default: 1.0
- **--model_name** - The name of Triton Inference Server model. Default: enhancer_ensemble
- **--batch_size** - The size of the batch of async tasks every job will process
- **--triton_address** - The Triton Inference Server address
- **--triton_port** - The Triton Inference Server port
- **--n_jobs** - Number of parallel jobs. If -1 specified, use all available CPU cores.

## Расстановка запятых и точек в местах пауз голоса с помощью Montreal Forced Aligner

Montreal Forced Aligner позволяет по имеющимся текстам и аудиофайлам получить разметку времени произнесения каждого слова/фонемы. Таким образом можно определить, насколько длительны паузы между словами. Используя эту информацию можно расставить знаки препинания для того, чтобы модель синтеза голоса выучила соответвтующие паузы.

### Установка зависимостей

К сожалению Montreal Forced Aligner (MFA) имеет специфические зависимости, которые не устанавливаются корректно с помощью pip. Однако есть готовый контейнер с треубетыми библиотеками.

Чтобы поднять соответствующий контейнер используйте:

```bash
docker run -it --name MFA_Processing -v [PATH_TO_DATA_TO_PROCESS]:/workspace/data -v $(pwd):/workspace mmcauliffe/montreal-forced-aligner
```

Дальшее нужно дополнительно установить библиотеку textgrid

```
pip install textgrid
```

### Обработка датасета с помощью MFA

Внутри поднятого контейнера запустите следующий скрипт

```bash
cd /workspace

python -m src.preprocessing.mfa_processing ./data/[YOUR_DATASET] 
```

Описание всех параметров представлено ниже:
- **input_path** - Path to data to process.
- **--n_jobs** - Number of parallel jobs to use while processing. -1 means to use all cores. Default: -1
- **--comma_duration** - Duration of pause which will be indicated with comma. Default: 0.15
- **--period_duration** - Duration of pause which will be indicated with period. Default: 0.3

## Распознование произнесенного текста с помощью ASR

Для оценки качества голоса в записях используется оценка с помощью Automatic Speech Recognition. Оригинальные текст сопоставляется с распознанным по WER и CER. В случаях, когда эти показатели превышают требуемый порог - информация о семплах удаляется из `metadata.csv`.

Для анализа используется Triton Inference Server с ASR моделью. Данный сервер находится под NDA. Принцип работы аналогичен [обработке Enhancer'ом](#улучшение-качества-с-помощью-resemble-enhancerа).

### Обработка датасета с помощью ASR

После поднятия ASR Triton Inference Server'a запустите следующий скрипт:

```
python -m src.preprocessing.asr_processing --dataset_path [PATH_TO_DATASET] --triton_port 127.0.0.1 --triton_port 9870 --cer_threshold 0.1 --wer_threshold 0.1
```

Описание всех параметров представлено ниже:
- **dataset_path** - Path to the dataset containing audio files.
- **--n_jobs** - Number of parallel jobs to use while processing. -1 means to use all cores. Default: -1
- **--wer_threshold** - WER threshold.. Default: 0.5
- **--cer_threshold** - CER threshold. Default: 0.5
- **--triton_address** - Address of the Triton Inference Server. Default: localhost
- **--triton_port** - Port of the Triton Inference Server. Default: 8000
- **--batch_size** - Batch size for processing audio files. Default: 10