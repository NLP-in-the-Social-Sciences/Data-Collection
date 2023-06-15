import nltk
import spacy
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')

stops = set(stopwords.words("english"))

def lemmatize_array(array)-> list: 
    return [lemmatize(paragraph) for paragraph in tqdm(array)]


def lemmatize(texts: str, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"], accuracy = "low") :
    """
    :param texts: aragraphs
    :param allowed_postags: allowed parts of speech
        return: 
            toekenized and lemmatized texts
    """
    if type(texts) != str: 
        raise Exception()
    
    if accuracy not in ["low", "high"]:
        raise Exception("In correct arguemt", accuracy)
    else:
        if accuracy != "low":
            accuracy = "high"

    #Use "python -m spacy download en_core_web_sm" in a terminal if error [E050]
    model = spacy.load("en_core_web_sm", disable=["parser","ner"])
    
    text_arr = texts.split(".")[:-1]
    sentece_arr = []

    for sentence in text_arr:
        tokens = model(sentence)
        tokens = " ".join([token.lemma_ for token in tokens
                           if token.pos_ in allowed_postags and token.lemma_ not in stops])
        
        sentece_arr.append(tokens)
    
    return ".".join(sentece_arr)

def gen_words(tokens: str):
    tokens = " ".join(simple_preprocess(tokens, deacc=True))
    
    return (tokens)


def main(): 

    paragraph = "My mum had me at 15 years. No idea who my dad is. I grew up with a single mum who would spend every last dollar on meth or coke. To say we were poor was an understatement. No amount of government assistance can get through to you if your mother is an addict. We moved around a lot, I went to 17 different schools growing up, having no food was a common occurrence. I've been homeless for periods of time as a kid. I've had to wash myself in public restrooms and from time to time I was sent to other 'relatives' to live. I was sexually abused on multiple occasions, and I've kept all of this to myself all these years. When you're a kid it's terrifying to speak out. You already live in a shaky, unstable world so uprooting the last foundation you have, even if it's a drug addled mother is unthinkable. Anyway, fast-forward. I tried really hard in school. I mean really hard. It was the only way I could see myself getting out of the hole I was in. My mum dropped out of school at 14 and all I knew is that I never wanted to end up like her. I got a job the day of my 15th birthday which in my country is the legal age you can start working and to this day, I'm 27 years now, I've not spent a day unemployed. I worked and saved as much as I could and when mum told me she was moving again when I was 16, I said no and moved out on my own. I was tired of starting over. I applied for emancipation and moved into a flat. I've seen her a handful of times since, I'm not even sure where she lives anymore.I went to university, kept up a 4.0 GPA while working near full-time and graduated with first class honors. I don't say this to brag. I sacrificed a lot to pull this off. I trashed my social life, never went on a holiday and ignored parts of my health because I wasn't a prodigy or anything close to it, I just fucking grinded my face off. I got a job in my field, got a post degree qualification, and in the last couple of years I've started clearing $100k+ per year."
        
    print(lemmatize(paragraph))

if __name__ == "__main__": 
    main()