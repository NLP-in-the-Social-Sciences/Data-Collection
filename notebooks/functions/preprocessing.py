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
    :param accuracy: (optional) accuracy needed from the model 
        return: 
            tokenized and lemmatized texts
    """
    if type(texts) != str: 
        raise Exception("No input string given.")
    
    if accuracy not in ["low", "high"]:
        raise Exception("In correct arguemt", accuracy)
    else:
        if accuracy != "high":
            #Use "python -m spacy download en_core_web_sm" in a terminal if error [E050]
            model = spacy.load("en_core_web_sm", disable=["parser","ner"])

        else:
            #Use "python -m spacy download 'en_core_web_trf'" in a terminal if error [E050]
            model = spacy.load('en_core_web_trf', disable=["parser","ner"])
    
    text_arr = texts.split(".")[:-1]
    sentece_arr = []

    for sentence in text_arr:
        tokens = model(sentence)
        tokens = " ".join([token.lemma_ for token in tokens
                        if token.pos_ in allowed_postags 
                        and token not in stops
                        and not token.is_punct
                        and not token.like_num
                        and not token.is_digit
                        and not token.is_space
                        and not token.is_currency]) # checking all this takes a lot of time
        
        sentece_arr.append(tokens)
    
    print("done")
    return " ".join(sentece_arr)

def gen_words(tokens: str):
    tokens = " ".join(simple_preprocess(tokens, deacc=True))
    
    return (tokens)

def main(): 
    # test
    paragraph = "In the tranquil meadows of a forgotten countryside, where time seemed to stretch its arms lazily across the horizon, a gentle breeze whispered secrets to the tall grass, swaying it in rhythmic undulations. The sun, ablaze with golden hues, cast its radiant beams upon the idyllic landscape, illuminating every blade of grass and infusing the air with a warm embrace. As the day unfolded, birds soared through the vast expanse of the sky, their wings outstretched in graceful arcs, painting fleeting patterns against the canvas of the heavens. Amidst this picturesque scene, a solitary figure, clad in a flowing cloak of vibrant colors, stood atop a hill, gazing at the panoramic vista that lay before them."
        
    print(lemmatize(paragraph,accuracy= "high"))

if __name__ == "__main__": 
    main()