# New Ethics Media Analysis

This repository contains Python code used for corpus-based linguistic analysis of the "new ethics" concept in Russian media, specifically from **Lenta.ru** and **Meduza.io** (2019–2024).

## Main dependencies

- nltk
- pymorphy2
- spacy
- gensim
- deeppavlov
- matplotlib

You can install the required libraries using:

```bash
pip install nltk pymorphy2 spacy gensim deeppavlov matplotlib
python -m spacy download ru_core_news_sm
```

## How to use

1. Prepare your own raw textual data.
2. Adjust file paths in the script `preprocessing_analysis.py` under `code/` directory.
3. Run the script to perform text preprocessing, frequency analysis, NER, and topic modeling.
