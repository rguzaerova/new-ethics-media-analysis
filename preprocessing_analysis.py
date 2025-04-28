# preprocessing_analysis.py

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy2
from collections import Counter
import spacy
from gensim import corpora
from gensim.models import LdaModel

# Cleaning and preprocessing
def clean_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text.lower()

def tokenize_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return tokens

def lemmatize_tokens(tokens):
    morph = pymorphy2.MorphAnalyzer()
    lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return lemmatized_tokens

# Frequency analysis
def save_word_frequencies(tokens, output_file):
    word_freq = Counter(tokens)
    with open(output_file, 'w', encoding='utf-8') as file:
        for word, freq in word_freq.most_common():
            file.write(f"{word}\t{freq}\n")
    return word_freq

# Named Entity Recognition
def named_entity_recognition(text):
    nlp = spacy.load("ru_core_news_sm")
    nlp.max_length = 1500000
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Topic modeling
def perform_topic_modeling(lemmatized_tokens, output_file, num_topics=5, num_words=10):
    texts = [lemmatized_tokens]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42)
    topics = lda_model.print_topics(num_topics=num_topics, num_words=num_words)
    with open(output_file, 'w', encoding='utf-8') as topic_file:
        for idx, topic in enumerate(topics):
            topic_file.write(f"Topic {idx + 1}: {topic}\n")

# Example usage (adjust paths accordingly)
# text = open('path_to_text.txt', 'r', encoding='utf-8').read()
# tokens = tokenize_text(clean_text(text))
# lemmatized_tokens = lemmatize_tokens(tokens)
# save_word_frequencies(lemmatized_tokens, 'word_frequencies.txt')
# entities = named_entity_recognition(text)
# perform_topic_modeling(lemmatized_tokens, 'topic_modeling.txt')
