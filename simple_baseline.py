import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import pytextrank

def clean_text(text):
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence.lower()) for sentence in sentences]
    stop_words = set(stopwords.words('english'))

    cleaned_words = [
        [word for word in sentence if word.isalpha() and word not in stop_words]
        for sentence in words
    ]

    return cleaned_words

def text_rank(input):
    en_nlp = spacy.load("en_core_web_sm")
    en_nlp.add_pipe("textrank", config={ "stopwords": { "word": ["NOUN"] } })
    doc = en_nlp(input)

    tr = doc._.textrank
    print(tr.elapsed_time)

    for sent in tr.summary(limit_phrases=10, limit_sentences=3):
        print(sent)

text = '''India recorded its lowest daily Covid-19 cases in over four months on Tuesday as it
registered 30,093 fresh cases of the coronavirus disease, the Union ministry of health and
family welfare data showed. The last time India's Covid-19 tally was below 30,000-mark was on 
March 16 when the country saw 28,903 fresh cases.
The country also saw 374 deaths due to Covid-19 in the last 24 hours, taking the death toll to 414,482. This is also the lowest death count India has seen after over three months. India witnessed deaths below 400 on March 30 when 354 fatalities were recorded.
Active cases of Covid-19 in the last 24 hours dipped sharply by 15,535, bringing the current infections in the country down to 406,130, the health ministry data showed. These account for 1.35% of the total infections reported in the country.
At least 45,254 people recovered from the infectious disease in the last 24 hours, taking India's recovery rate to 97.32%.'''

text_rank(text)