# transformers_overall

Проект по трансформерам на задаче классификации тональности текста.

Используется датасет SST-2 с двумя классами:

* `0` - negative;
* `1` - positive.

Основная модель: `distilbert-base-uncased`.

## Этапы

### День 1 - Токенизация

* загрузка `AutoTokenizer`;
* преобразование текста в токены и `input_ids`;
* работа с `attention_mask`, padding и truncation;
* токенизация батчей;
* анализ специальных токенов.

### День 2 - Hidden states

* загрузка `AutoModel`;
* получение `last_hidden_state`;
* извлечение CLS-эмбеддингов;
* получение эмбеддингов текстов батчами;
* вычисление cosine similarity.

### День 3 - Attention

* получение attention weights;
* анализ формы attention-матриц;
* сравнение разных слоёв и голов;
* визуализация attention через heatmap;
* использование общей цветовой шкалы;
* усреднение attention по головам.

### День 4 - Baseline

* подготовка сбалансированной выборки SST-2;
* извлечение CLS-эмбеддингов с помощью замороженного DistilBERT;
* обучение Logistic Regression;
* оценка через classification report, accuracy и macro F1;
* сохранение baseline модели и фиксированных train/validation выборок.

### День 5 - Fine-tuning

* создание `Dataset` и `DataLoader`;
* загрузка `AutoModelForSequenceClassification`;
* fine-tuning `distilbert-base-uncased`;
* обучение в течение 3 эпох;
* оценка на validation выборке;
* сохранение лучшей модели по validation macro F1;
* сохранение истории обучения и итоговых метрик.

### День 6 - Инференс и сравнение

* загрузка baseline и fine-tuned моделей;
* реализация `predict_fine_tuned`;
* реализация `predict_baseline`;
* batch inference;
* сравнение моделей на одинаковой validation выборке;
* построение confusion matrix для обеих моделей;
* сравнение accuracy и macro F1;
* сохранение итогового отчёта.

### День 7 - Анализ ошибок и демо

* поиск False Positive и False Negative;
* анализ ошибочных предсказаний и уверенности модели;
* сравнение длины ошибочных и всех текстов;
* сохранение полного анализа ошибок;
* создание Gradio-приложения;
* локальный inference через fine-tuned DistilBERT.

## Результаты

Данные:

* train: 1600 текстов;
* validation: 400 текстов;
* классы: negative и positive;
* размер CLS-эмбеддинга baseline: 768.

### Baseline

* Accuracy: `0.8525`;
* Macro F1: `0.8525`.

### Fine-tuned DistilBERT

* Accuracy: `0.8900`;
* Macro F1: `0.8900`;
* абсолютное улучшение Macro F1: `0.0375`;
* относительное улучшение Macro F1: `4.40%`.

Fine-tuned модель допустила 44 ошибки на 400 validation-примерах:

* False Positives: 19;
* False Negatives: 25.

Полные результаты находятся в:

```text
outputs/baseline/baseline_results.txt
outputs/fine_tuned/fine_tuned_results.txt
outputs/comparison/comparison_results.txt
outputs/error_analysis/error_analysis.txt
```

## Клонирование и окружение

Для хранения весов fine-tuned модели используется Git LFS.

Перед клонированием нужно убедиться, что Git LFS установлен:

```bash
git lfs install
```

Клонирование репозитория:

```bash
git clone https://github.com/Agito-dm/transformers_overall.git
cd transformers_overall
```

При обычном клонировании Git LFS загружает веса автоматически.
При необходимости их можно получить вручную:

```bash
git lfs pull
```

Создание Conda-окружения:

```bash
conda create -n transformers_overall python=3.10
conda activate transformers_overall
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Добавление окружения как Jupyter kernel:

```bash
python -m ipykernel install --user --name transformers_overall --display-name "Python (transformers_overall)"
```

Для работы с notebook в VS Code необходимо выбрать kernel:

```text
Python (transformers_overall)
```

## Модели

Проект содержит обязательные артефакты моделей:

```text
models/
  baseline/
    logistic_regression.joblib

  fine_tuned_model/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
```

`model.safetensors` хранится через Git LFS.

Baseline-модель используется в Дне 6 для сравнения, а fine-tuned модель используется в Днях 6–7 и Gradio-приложении.

## Запуск Gradio-приложения

Из корня проекта нужно выполнить:

```bash
conda activate transformers_overall
python app.py
```

После запуска интерфейс будет доступен по адресу:

```text
http://127.0.0.1:7860
```

Приложение принимает английский текст и выводит вероятности классов `Negative` и `Positive`.

## Структура проекта

```text
transformers_overall/
  data/
    sst2_sample.csv
    splits/
      train.csv
      validation.csv

  models/
    baseline/
      logistic_regression.joblib
    fine_tuned_model/
      config.json
      model.safetensors
      tokenizer.json
      tokenizer_config.json

  notebooks/
    transformers_day01.ipynb
    transformers_day02.ipynb
    transformers_day03.ipynb
    transformers_day04.ipynb
    transformers_day05.ipynb
    transformers_day06.ipynb
    transformers_day07.ipynb

  outputs/
    attention/

    baseline/
      baseline_results.txt

    fine_tuned/
      fine_tuned_results.txt
      training_history.csv

    comparison/
      comparison_results.txt
      confusion_matrix_baseline.png
      confusion_matrix_fine_tuned.png

    error_analysis/
      error_analysis.txt
      error_examples.csv

  app.py
  .gitignore
  README.md
  requirements.txt
```
