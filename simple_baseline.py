import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


def clean_text(text):
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence.lower()) for sentence in sentences]
    stop_words = set(stopwords.words('english'))

    cleaned_words = [
        [word for word in sentence if word.isalpha() and word not in stop_words]
        for sentence in words
    ]

    return cleaned_words
