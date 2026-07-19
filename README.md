# transformers_overall

Проект по трансформерам на задаче классификации тональности текста

## Этапы

### День 1 — Токенизация

- загрузка `AutoTokenizer`;
- преобразование текста в токены и `input_ids`;
- работа с `attention_mask`, padding и truncation;
- токенизация батчей;
- анализ специальных токенов.

### День 2 — Hidden states

- загрузка `AutoModel`;
- получение `last_hidden_state`;
- извлечение CLS-эмбеддингов;
- получение эмбеддингов текстов батчами;
- вычисление cosine similarity.

### День 3 — Attention

- получение attention weights;
- анализ формы attention-матриц;
- сравнение разных слоёв и голов;
- визуализация attention через heatmap;
- использование общей цветовой шкалы;
- усреднение attention по головам.

### День 4 — Baseline

- подготовка сбалансированной выборки SST-2;
- извлечение CLS-эмбеддингов с помощью замороженного DistilBERT;
- обучение Logistic Regression;
- оценка через classification report, accuracy и macro F1.

## Результат baseline

Модель: `distilbert-base-uncased`

Данные:

- train: 1600 текстов;
- test: 400 текстов;
- классы: negative и positive;
- размер CLS-эмбеддинга: 768.

Метрики:

- Accuracy: `0.8525`;
- Macro F1: `0.8525`.

Полный отчёт находится в:

```text
outputs/baseline/baseline_results.txt
```

Структура проекта
```text
transformers_overall/
├── data/
│   └── sst2_sample.csv
├── notebooks/
│   ├── transformers_day01.ipynb
│   ├── transformers_day02.ipynb
│   ├── transformers_day03.ipynb
│   └── transformers_day04.ipynb
├── outputs/
│   ├── attention/
│   └── baseline/
│       └── baseline_results.txt
├── .gitignore
├── README.md
└── requirements.txt
```
