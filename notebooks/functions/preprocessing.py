import nltk
import spacy
import numpy as np
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')

stops = set(stopwords.words("english"))

def lemmatize(texts: str, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]) :

    #Use "python -m spacy download en_core_web_sm" in a terminal if error [E050]
    model = spacy.load("en_core_web_sm", disable=["parser","ner"])

    tokens = model(texts)
    tokens = " ".join([token.lemma_ for token in tokens
                    if token.pos_ in allowed_postags and token.lemma_ not in stops])
    
    return tokens

def gen_words(tokens: str):
    tokens = " ".join(simple_preprocess(tokens, deacc=True))
    
    return (tokens)
